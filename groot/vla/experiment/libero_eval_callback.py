"""Inline LIBERO simulation evaluation callback for DreamZero training.

Runs LIBERO rollouts using the training model directly (no checkpoint loading).
Rank 0 runs the LIBERO environment; inference runs on rank 0 only (valid for
ZeRO-2 which replicates full model weights on every GPU).

Usage: add a LiberoEvalCallback to the trainer via create_trainer:
    trainer.add_callback(LiberoEvalCallback(
        task_suite_name="libero_10",
        num_trials_per_task=3,
        exp_cfg_dir=exp_cfg_dir,
    ))
"""

from __future__ import annotations

import collections
import json
import logging
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf, open_dict
from transformers import TrainerCallback


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to axis-angle (3D)."""
    # Clip for numerical stability
    quat = np.array(quat, dtype=np.float64)
    # Normalize
    quat = quat / (np.linalg.norm(quat) + 1e-9)
    x, y, z, w = quat
    # angle
    angle = 2.0 * np.arccos(np.clip(abs(w), 0.0, 1.0))
    s = np.sqrt(1.0 - w * w)
    if s < 1e-6:
        return np.array([0.0, 0.0, 0.0])
    axis = np.array([x, y, z]) / s
    return axis * angle

logger = logging.getLogger(__name__)

_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


# ---------------------------------------------------------------------------
# Lightweight policy wrapper – uses training model without loading from disk
# ---------------------------------------------------------------------------

class _InlinePolicy:
    """Wraps the (DeepSpeed-unwrapped) training model for LIBERO inference.

    Replicates only the parts of GrootSimPolicy needed for inference:
      - eval_transform  (obs normalisation / action unnormalisation)
      - lazy_joint_forward_causal  (transform → model → untransform)
    """

    def __init__(
        self,
        trained_model,
        exp_cfg_dir: Path,
        embodiment_tag: str,
        device: str | int,
        eval_bf16: bool = True,
    ):
        from hydra.utils import instantiate
        from groot.vla.data.schema import DatasetMetadata
        from tianshou.data import Batch

        self.trained_model = trained_model
        self.device = device
        self.eval_bf16 = eval_bf16
        self.Batch = Batch

        # Load training config and metadata from the experiment directory.
        exp_cfg_dir = Path(exp_cfg_dir)
        train_cfg = OmegaConf.load(exp_cfg_dir / "conf.yaml")
        self.train_cfg = train_cfg

        with open(exp_cfg_dir / "metadata.json") as f:
            metadatas = json.load(f)
        metadata = DatasetMetadata.model_validate(metadatas[embodiment_tag])

        # Build eval transform (same as GrootSimPolicy does).
        eval_transform_cfg = train_cfg.transforms[embodiment_tag]
        # Drop augmentation transforms (random crop, colour jitter) for eval.
        skipped = {
            "groot.vla.data.transform.VideoCrop",
            "groot.vla.data.transform.VideoColorJitter",
        }
        with open_dict(eval_transform_cfg):
            eval_transform_cfg.transforms = [
                t for t in eval_transform_cfg.transforms
                if t._target_ not in skipped
            ]
        eval_transform = instantiate(eval_transform_cfg)
        eval_transform.set_metadata(metadata)
        eval_transform.eval()
        self.eval_transform = eval_transform
        self.relative_action = train_cfg.get("relative_action", False)
        self.relative_action_per_horizon = train_cfg.get("relative_action_per_horizon", False)
        self.relative_action_keys = train_cfg.get("relative_action_keys", [])

    def infer(self, obs: dict):
        """Run one inference step.

        Args:
            obs: raw observation dict with numpy arrays (same format as
                 dreamzero_eval.py's dz_obs).

        Returns:
            action_chunk: np.ndarray of shape (H, 7)
        """
        from tianshou.data import Batch

        # Add batch dim if absent.
        obs_batched = {k: (v[None] if isinstance(v, np.ndarray) and v.ndim == len(v.shape)
                           else v)
                       for k, v in obs.items()}

        normalized_input = self.eval_transform(obs_batched)
        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        # Cast to bf16 if needed.
        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        with torch.inference_mode():
            model_pred = self.trained_model.lazy_joint_video_action_causal(normalized_input)

        normalized_action = model_pred["action_pred"].float()  # (1, H, D)

        # Unnormalise actions.
        unnormalized = self.eval_transform.unapply(
            dict(action=normalized_action.cpu())
        )

        # Relative → absolute action conversion.
        if (self.relative_action or self.relative_action_per_horizon) and self.relative_action_keys:
            for key in self.relative_action_keys:
                action_key = f"action.{key}"
                state_key = f"state.{key}"
                if action_key not in unnormalized:
                    continue
                last_state = obs_batched.get(state_key)
                if last_state is None:
                    continue
                if isinstance(last_state, np.ndarray):
                    last_state = last_state[..., -1, :] if last_state.ndim >= 2 else last_state
                unnorm = unnormalized[action_key]
                if unnorm.ndim > last_state.ndim:
                    last_state = np.expand_dims(last_state, axis=-2)
                unnormalized[action_key] = unnorm + last_state

        # Merge joint + gripper → (H, 7) array.
        joint = unnormalized.get("action.joint_position")  # (1, H, 6)
        grip = unnormalized.get("action.gripper_position")  # (1, H, 1)
        if joint is not None and grip is not None:
            joint = np.asarray(joint)[0]  # (H, 6)
            grip = np.asarray(grip)[0]    # (H, 1)
            return np.concatenate([joint, grip], axis=-1)  # (H, 7)
        # Fallback: return whatever we got
        return next(iter(unnormalized.values()))


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class LiberoEvalCallback(TrainerCallback):
    """Evaluate on LIBERO simulation at training start and/or after each save.

    Args:
        task_suite_name: LIBERO benchmark name, e.g. "libero_10".
        num_trials_per_task: rollouts per task (keep low for quick eval, e.g. 3).
        exp_cfg_dir: path to the experiment config dir (holds conf.yaml +
                     metadata.json).  Populated automatically by BaseExperiment.
        embodiment_tag: dataset embodiment tag; default "libero_sim".
        seed: RNG seed for LIBERO env resets.
        replan_steps: execute this many actions from each chunk before replanning.
        eval_on_save: also evaluate after every checkpoint save (in addition to
                      the step-0 eval triggered by eval_on_start).
        num_steps_wait: dummy steps at episode start before querying the policy.
        max_tasks: cap how many tasks to evaluate (None = all tasks in suite).
    """

    def __init__(
        self,
        task_suite_name: str,
        num_trials_per_task: int,
        exp_cfg_dir: Path | str,
        embodiment_tag: str = "libero_sim",
        seed: int = 0,
        replan_steps: int = 8,
        eval_on_save: bool = False,
        eval_on_train_begin: bool = True,
        num_steps_wait: int = 10,
        max_tasks: Optional[int] = None,
        max_steps_per_episode: Optional[int] = None,
    ):
        self.task_suite_name = task_suite_name
        self.num_trials_per_task = num_trials_per_task
        self.exp_cfg_dir = Path(exp_cfg_dir)
        self.embodiment_tag = embodiment_tag
        self.seed = seed
        self.replan_steps = replan_steps
        self.eval_on_save = eval_on_save
        self.eval_on_train_begin = eval_on_train_begin
        self.num_steps_wait = num_steps_wait
        self.max_tasks = max_tasks
        self.max_steps_per_episode = max_steps_per_episode

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self.eval_on_train_begin:
            return
        self._run_eval(model, state, tag="step0")

    def on_save(self, args, state, control, model=None, **kwargs):
        if self.eval_on_save:
            self._run_eval(model, state, tag=f"step{state.global_step}")

    # ------------------------------------------------------------------
    # Core eval logic
    # ------------------------------------------------------------------

    def _run_eval(self, model, state, tag: str = ""):
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Sync all ranks before switching mode.
        if dist.is_initialized():
            dist.barrier()

        model.eval()
        results = None
        try:
            if rank == 0:
                # Unwrap DeepSpeed / DDP to get the raw nn.Module.
                raw_model = model.module if hasattr(model, "module") else model
                device = next(raw_model.parameters()).device

                policy = _InlinePolicy(
                    trained_model=raw_model,
                    exp_cfg_dir=self.exp_cfg_dir,
                    embodiment_tag=self.embodiment_tag,
                    device=device,
                    eval_bf16=True,
                )
                with torch.no_grad():
                    results = self._libero_loop(policy)

                self._log_results(results, state, tag)
        except Exception:
            logger.error("LiberoEvalCallback error:\n%s", traceback.format_exc())
        finally:
            torch.cuda.empty_cache()
            model.train()
            if dist.is_initialized():
                dist.barrier()

    def _libero_loop(self, policy: _InlinePolicy) -> dict:
        """Run LIBERO rollouts on rank 0. Returns a summary dict."""
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[self.task_suite_name]()
        n_tasks = suite.n_tasks
        if self.max_tasks is not None:
            n_tasks = min(n_tasks, self.max_tasks)
        max_steps = _MAX_STEPS.get(self.task_suite_name, 400)
        if self.max_steps_per_episode is not None:
            max_steps = min(max_steps, self.max_steps_per_episode)
        np.random.seed(self.seed)

        total_eps = total_succ = 0
        per_task = []

        for task_id in range(n_tasks):
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=224,
                camera_widths=224,
            )
            env.seed(self.seed)
            desc = str(task.language)

            task_succ = task_eps = 0
            for ep_idx in range(self.num_trials_per_task):
                env.reset()
                obs = env.set_init_state(init_states[ep_idx % len(init_states)])

                action_plan: collections.deque = collections.deque()
                t = 0
                done = False
                reward = 0.0

                while t < max_steps + self.num_steps_wait:
                    if t < self.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    if not action_plan:
                        img   = np.ascontiguousarray(obs["agentview_image"])
                        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
                        # state.joint_position = EEF pose (xyz + axis-angle rot) = 6-dim
                        # state.gripper_position = gripper qpos (2 fingers) = 2-dim
                        # This matches the training data layout in observation.state[0:6] / [6:8]
                        eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64)      # (3,)
                        eef_rot = _quat2axisangle(np.array(obs["robot0_eef_quat"], dtype=np.float64))  # (3,)
                        dz_obs = {
                            "video.agentview_rgb":   img[None].astype(np.uint8),
                            "video.eye_in_hand_rgb": wrist[None].astype(np.uint8),
                            "state.joint_position":
                                np.concatenate([eef_pos, eef_rot]).reshape(1, -1),       # (1, 6)
                            "state.gripper_position":
                                np.array(obs["robot0_gripper_qpos"],
                                         dtype=np.float64).reshape(1, -1),               # (1, 2)
                            "annotation.language.language_instruction": desc,
                        }
                        try:
                            chunk = policy.infer(dz_obs)  # (H, 7)
                            n_exec = min(self.replan_steps, len(chunk))
                            action_plan.extend(chunk[:n_exec])
                        except Exception:
                            logger.error("Inference error:\n%s", traceback.format_exc())
                            break

                    action = action_plan.popleft()
                    obs, reward, done, _ = env.step(action.tolist())
                    t += 1
                    if done and reward > 0:
                        task_succ += 1
                        total_succ += 1
                        break

                torch.cuda.empty_cache()
                task_eps += 1
                total_eps += 1
                success = bool(done and reward > 0)
                logger.info(
                    "  [%s] task %d ep %d/%d: %s",
                    self.task_suite_name, task_id,
                    ep_idx + 1, self.num_trials_per_task,
                    "SUCCESS" if success else "FAILURE",
                )

            env.close()
            rate = task_succ / task_eps if task_eps else 0.0
            per_task.append({
                "task_id": task_id,
                "description": desc,
                "success_rate": rate,
                "successes": task_succ,
                "trials": task_eps,
            })
            logger.info(
                "[%s] task %d: %d/%d = %.1f%%  [%s]",
                self.task_suite_name, task_id,
                task_succ, task_eps, rate * 100, desc,
            )

        total_rate = total_succ / total_eps if total_eps else 0.0
        logger.info(
            "\n%s\n%s: %d/%d = %.1f%%\n%s",
            "=" * 60, self.task_suite_name,
            total_succ, total_eps, total_rate * 100,
            "=" * 60,
        )
        return {
            "success_rate": total_rate,
            "total_successes": total_succ,
            "total_episodes": total_eps,
            "per_task": per_task,
        }

    def _log_results(self, results: dict, state, tag: str):
        try:
            import wandb
            if wandb.run is not None:
                step = state.global_step
                suite = self.task_suite_name
                wandb.log(
                    {
                        f"libero_eval/{suite}/success_rate": results["success_rate"],
                        f"libero_eval/{suite}/total_successes": results["total_successes"],
                        f"libero_eval/{suite}/total_episodes": results["total_episodes"],
                    },
                    step=step,
                )
                # Per-task breakdown.
                for pt in results["per_task"]:
                    wandb.log(
                        {f"libero_eval/{suite}/task{pt['task_id']}_success_rate": pt["success_rate"]},
                        step=step,
                    )
        except Exception:
            pass
        logger.info(
            "LiberoEval [%s] %s: success_rate=%.3f (%d/%d)",
            tag, self.task_suite_name,
            results["success_rate"],
            results["total_successes"],
            results["total_episodes"],
        )
