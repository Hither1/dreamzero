"""
Convert LIBERO demonstration datasets (HDF5) to LeRobot v2.0 format
in a way that aligns with DreamZero/VLA finetuning expectations.

This version INFERS state/action dims + observation keys from the dataset,
instead of hard-coding them.

It supports both common LIBERO HDF5 schemas:

A) "robot0_*" schema (matches many Diffusion Policy loaders):
   - obs/robot0_eef_pos          (T, 3)
   - obs/robot0_eef_quat         (T, 4) quaternion [x,y,z,w]
   - obs/robot0_gripper_qpos     (T, G) typically G=2  -> yields state dim 3+3+2 = 8
   - obs/agentview_image         (T, H, W, 3)
   - obs/robot0_eye_in_hand_image(T, H, W, 3)
   - actions                     (T, A) typically A=7

B) Older schema:
   - obs/ee_states               (T, 6) already [pos(3), axis-angle(3)]
   - obs/gripper_states          (T, G) (often 1)
   - obs/agentview_rgb, obs/eye_in_hand_rgb
   - actions                     (T, A)

Output columns (parquet):
- observation.state : concatenated float64 (T, state_dim) = [eef(6), gripper(G)]
- action            : float64 (T, action_dim)
- annotation.language.language_instruction : int64 per-frame task index
- task_index        : int64 per-frame task index (copied from language annotation)
- timestamp, episode_index, frame_index

Optional debug columns (enabled with --write-debug-columns):
- state.joint_position, state.gripper_position
- action.joint_position, action.gripper_position

Output structure:
  <output_dir>/
    meta/modality.json
    meta/episodes.jsonl
    meta/tasks.jsonl
    meta/info.json
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.agentview_rgb/episode_000000.mp4
    videos/chunk-000/observation.images.eye_in_hand_rgb/episode_000000.mp4

Usage (fast, reads from HDF5 directly):
  python scripts/data/convert_libero.py \
      /path/to/libero_split_dir \
      /path/to/output_dir \
      --fps 20 -n 8

Usage (re-renders via LIBERO environment replay):
  python scripts/data/convert_libero.py \
      /path/to/libero_split_dir \
      /path/to/output_dir \
      --resolution 224 \
      --task_suite_name libero_10 \
      --fps 20
  Note: re-rendering is single-threaded (MuJoCo envs cannot be forked).
"""

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import av
import h5py
import numpy as np
import polars as pl
import tqdm


# Avoid thread oversubscription when using multiprocessing + numpy/ffmpeg
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# Output video keys (canonical names used in LeRobot meta + paths)
OUTPUT_IMAGE_KEYS = ["agentview_rgb", "eye_in_hand_rgb"]
LANG_KEYS = ["language_instruction"]
CHUNKS_SIZE = 1000


@dataclass(frozen=True)
class LiberoSchema:
    # Which observation style we have
    uses_robot0_keys: bool

    # Image dataset names inside demo["obs"] (source keys)
    agentview_src_key: str
    eye_in_hand_src_key: str

    # Action dim
    action_dim: int

    # State composition
    eef_dim: int           # usually 6 (pos3 + axisangle3)
    gripper_dim: int       # 1 or 2 typically
    obs_state_dim: int     # eef_dim + gripper_dim

    # How we will split action for debug columns (optional)
    eef_action_dim: int
    gripper_action_dim: int


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to axis-angle (3D), matching robosuite convention."""
    quat = quat.astype(np.float64)
    quat = quat / np.linalg.norm(quat)
    denom = np.sqrt(max(1.0 - quat[3] ** 2, 0.0))
    if denom < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] / denom) * 2.0 * np.arccos(np.clip(quat[3], -1.0, 1.0))


def encode_video(frames: np.ndarray, output_path: Path, fps: int) -> None:
    """Encode (T, H, W, 3) uint8 RGB array to h264 MP4 via PyAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    options = {
        "threads": "1",
        "thread_type": "slice",
        "preset": "ultrafast",
        "tune": "zerolatency",
        "crf": "23",
    }

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames shape (T,H,W,3), got {frames.shape}")

    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=fps, options=options)
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"

    video_frame = av.VideoFrame(width=stream.width, height=stream.height, format="rgb24")
    frame_array = video_frame.to_ndarray(format="rgb24")

    for frame in frames:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        frame_array[:] = frame
        for packet in stream.encode(video_frame):
            container.mux(packet)

    for packet in stream.encode(None):
        container.mux(packet)

    container.close()


def _concat_info(keys: List[str], dims: List[int]) -> Dict[str, Dict[str, int]]:
    """Build concat_info like convert_droid.py uses for modality.json."""
    assert len(keys) == len(dims)
    out: Dict[str, Dict[str, int]] = {}
    start = 0
    for k, d in zip(keys, dims):
        out[k] = {"start": start, "end": start + int(d)}
        start += int(d)
    return out


def infer_libero_schema(hdf5_path: Path) -> LiberoSchema:
    """
    Infer obs/action schema from the first demo in an HDF5 file.
    Supports both:
      - Robot0 keys: obs/robot0_eef_pos, obs/robot0_eef_quat, obs/robot0_gripper_qpos,
        obs/agentview_image, obs/robot0_eye_in_hand_image
      - Older keys: obs/ee_states, obs/gripper_states, obs/agentview_rgb, obs/eye_in_hand_rgb
    """
    with h5py.File(hdf5_path, "r") as f:
        demo_key = sorted(list(f["data"].keys()))[0]
        demo = f["data"][demo_key]
        obs = demo["obs"]

        action_dim = int(demo["actions"].shape[1])

        uses_robot0 = ("robot0_eef_pos" in obs) and ("robot0_eef_quat" in obs)

        if uses_robot0:
            agentview_src_key = "agentview_image" if "agentview_image" in obs else "agentview_rgb"
            eye_src_key = "robot0_eye_in_hand_image" if "robot0_eye_in_hand_image" in obs else "eye_in_hand_rgb"

            if "robot0_gripper_qpos" in obs:
                g = obs["robot0_gripper_qpos"]
                gripper_dim = int(g.shape[1]) if g.ndim == 2 else 1
            elif "gripper_states" in obs:
                gripper_dim = int(obs["gripper_states"].shape[1])
            else:
                raise KeyError(
                    "Could not find gripper observations (robot0_gripper_qpos or gripper_states)."
                )

            eef_dim = 6  # pos(3) + axis-angle(3)
        else:
            # Older convention
            if "ee_states" not in obs:
                raise KeyError("Expected obs/ee_states for non-robot0 schema.")
            if "gripper_states" not in obs:
                raise KeyError("Expected obs/gripper_states for non-robot0 schema.")
            if "agentview_rgb" not in obs or "eye_in_hand_rgb" not in obs:
                raise KeyError("Expected obs/agentview_rgb and obs/eye_in_hand_rgb for non-robot0 schema.")

            agentview_src_key = "agentview_rgb"
            eye_src_key = "eye_in_hand_rgb"
            eef_dim = int(obs["ee_states"].shape[1])
            gripper_dim = int(obs["gripper_states"].shape[1])

        obs_state_dim = int(eef_dim + gripper_dim)

        # Action split heuristic: last dim is gripper if action_dim >= 1
        gripper_action_dim = 1 if action_dim >= 1 else 0
        eef_action_dim = int(action_dim - gripper_action_dim)

        return LiberoSchema(
            uses_robot0_keys=uses_robot0,
            agentview_src_key=agentview_src_key,
            eye_in_hand_src_key=eye_src_key,
            action_dim=action_dim,
            eef_dim=int(eef_dim),
            gripper_dim=int(gripper_dim),
            obs_state_dim=obs_state_dim,
            eef_action_dim=eef_action_dim,
            gripper_action_dim=int(gripper_action_dim),
        )


def make_dummy_action(action_dim: int) -> List[float]:
    """No-op action for settling physics; keep 'gripper closed' as last dim if present."""
    if action_dim <= 0:
        return []
    a = [0.0] * action_dim
    a[-1] = -1.0
    return a


def _read_demo_arrays(
    hdf5_path: Path, demo_key: str, schema: LiberoSchema
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      actions: (T, action_dim)
      eef_states: (T, eef_dim) — usually [eef_pos(3), eef_axisangle(3)]
      gripper_states: (T, gripper_dim)
      agentview: (T, H, W, 3)
      eye_in_hand: (T, H, W, 3)
    """
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_key]
        obs = demo["obs"]

        actions = demo["actions"][:]  # (T, action_dim)

        if schema.uses_robot0_keys:
            # Match Diffusion Policy loader:
            # eef_pos: (T,3), eef_quat: (T,4) -> axis-angle (T,3)
            eef_pos = obs["robot0_eef_pos"][:].astype(np.float64)
            eef_quat = obs["robot0_eef_quat"][:].astype(np.float64)
            eef_axisangle = np.stack([_quat2axisangle(q) for q in eef_quat]).astype(np.float64)
            eef_states = np.concatenate([eef_pos, eef_axisangle], axis=1)  # (T,6)

            if "robot0_gripper_qpos" in obs:
                gripper_states = obs["robot0_gripper_qpos"][:].astype(np.float64)
                if gripper_states.ndim == 1:
                    gripper_states = gripper_states.reshape(-1, 1)
            else:
                # fallback
                gripper_states = obs["gripper_states"][:].astype(np.float64)
                if gripper_states.ndim == 1:
                    gripper_states = gripper_states.reshape(-1, 1)

            agentview = obs[schema.agentview_src_key][:]
            eye_in_hand = obs[schema.eye_in_hand_src_key][:]
        else:
            # Older convention: ee_states already axis-angle
            eef_states = obs["ee_states"][:].astype(np.float64)

            gripper_states = obs["gripper_states"][:].astype(np.float64)
            if gripper_states.ndim == 1:
                gripper_states = gripper_states.reshape(-1, 1)

            agentview = obs["agentview_rgb"][:]
            eye_in_hand = obs["eye_in_hand_rgb"][:]

    return actions, eef_states, gripper_states, agentview, eye_in_hand


def _render_demo_arrays(
    hdf5_path: Path, demo_key: str, bddl_file: Path, resolution: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Re-render a demo at `resolution` by replaying actions in the LIBERO environment.

    Loads the full MuJoCo simulation state from the HDF5 to restore the exact initial
    configuration, then replays every stored action and records the resulting observations
    at the requested camera resolution.

    Returns:
      actions: (T, action_dim)
      eef_states: (T, 6)  — [eef_pos(3), eef_axisangle(3)]
      gripper_states: (T, G)  — robot0_gripper_qpos (often G=2) if available, else (T,1)
      agentview: (T, resolution, resolution, 3)
      eye_in_hand: (T, resolution, resolution, 3)
    """
    from libero.libero.envs import OffScreenRenderEnv

    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_key]
        orig_actions = demo["actions"][:]  # (T, action_dim)
        orig_states = demo["states"][:]    # (T, state_dim) full MuJoCo state

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)

    try:
        env.reset()
        obs = env.set_init_state(orig_states[0])

        dummy = make_dummy_action(int(orig_actions.shape[1]))

        # Let physics settle before recording
        for _ in range(10):
            obs, _, _, _ = env.step(dummy)

        agentview_imgs = []
        eye_in_hand_imgs = []
        eef_states_out = []
        gripper_states_out = []

        for action in orig_actions:
            # Record observation BEFORE taking the action (matches common HDF5 convention)
            # Images
            agentview_imgs.append(obs.get("agentview_image", obs.get("agentview_rgb")))
            eye_in_hand_imgs.append(obs.get("robot0_eye_in_hand_image", obs.get("eye_in_hand_rgb")))

            # EEF
            eef_pos = obs["robot0_eef_pos"]          # (3,)
            eef_quat = obs["robot0_eef_quat"]        # (4,) [x,y,z,w]
            eef_axisangle = _quat2axisangle(np.asarray(eef_quat))  # (3,)
            eef_states_out.append(np.concatenate([np.asarray(eef_pos), eef_axisangle]).astype(np.float64))

            # Gripper
            if "robot0_gripper_qpos" in obs:
                gqpos = np.asarray(obs["robot0_gripper_qpos"]).astype(np.float64)
                if gqpos.ndim == 0:
                    gqpos = gqpos.reshape(1)
                gripper_states_out.append(gqpos.reshape(-1))
            else:
                # fallback to something 1D
                gqpos = obs.get("gripper_qpos", np.array([-1.0], dtype=np.float64))
                gqpos = np.asarray(gqpos).astype(np.float64).reshape(-1)
                gripper_states_out.append(gqpos[:1])

            obs, _, _, _ = env.step(action.tolist())

    finally:
        env.close()

    gripper_arr = np.stack([g.reshape(1, -1) if g.ndim == 1 else g for g in gripper_states_out], axis=0)
    gripper_arr = gripper_arr.reshape(len(gripper_states_out), -1).astype(np.float64)

    return (
        orig_actions,
        np.stack(eef_states_out).astype(np.float64),
        gripper_arr,
        np.stack(agentview_imgs),
        np.stack(eye_in_hand_imgs),
    )


def _get_bddl_file(hdf5_path: Path, task_suite_name: str = None) -> Path:
    """
    Look up the BDDL file for a task.

    Tries (in order):
      1. benchmark lookup via task_suite_name + task name from filename
      2. env_args JSON attribute stored in the HDF5 file
    """
    from libero.libero import get_libero_path

    # Try benchmark lookup first
    if task_suite_name is not None:
        from libero.libero import benchmark
        task_name = hdf5_path.stem.replace("_demo", "")
        benchmark_dict = benchmark.get_benchmark_dict()
        suite = benchmark_dict[task_suite_name]()
        for i in range(suite.n_tasks):
            task = suite.get_task(i)
            if task.name == task_name:
                return Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        raise ValueError(f"Task '{task_name}' not found in suite '{task_suite_name}'")

    # Fallback: read from HDF5 env_args attribute
    with h5py.File(hdf5_path, "r") as f:
        env_args_raw = f["data"].attrs.get("env_args", None)
        if env_args_raw is None:
            raise ValueError(
                f"No env_args attribute in {hdf5_path} and --task_suite_name not provided. "
                "Pass --task_suite_name to specify the task suite for BDDL lookup."
            )
        env_args = json.loads(env_args_raw)
        bddl_path = Path(env_args["bddl_file_name"])
        if not bddl_path.exists():
            # Try relative to libero bddl_files root
            bddl_path = Path(get_libero_path("bddl_files")) / bddl_path.name
        return bddl_path


def process_episode(
    ep_idx: int,
    hdf5_path: Path,
    demo_key: str,
    task_name: str,
    task_idx: int,
    output_path: Path,
    fps: int,
    write_debug_columns: bool,
    schema: LiberoSchema,
    resolution: int = 224,
    bddl_file: Path = None,
) -> Dict:
    """Convert one LIBERO demo to parquet + videos. Returns episode metadata dict."""
    chunk_idx = ep_idx // CHUNKS_SIZE

    (output_path / f"data/chunk-{chunk_idx:03d}").mkdir(parents=True, exist_ok=True)
    for img_key in OUTPUT_IMAGE_KEYS:
        (output_path / f"videos/chunk-{chunk_idx:03d}/observation.images.{img_key}").mkdir(
            parents=True, exist_ok=True
        )

    if resolution == 128 or bddl_file is None:
        actions, eef_states, gripper_states, agentview, eye_in_hand = _read_demo_arrays(
            hdf5_path, demo_key, schema
        )
    else:
        actions, eef_states, gripper_states, agentview, eye_in_hand = _render_demo_arrays(
            hdf5_path, demo_key, bddl_file, resolution
        )

    T = int(actions.shape[0])
    if eef_states.shape[0] != T or gripper_states.shape[0] != T:
        raise ValueError(
            f"Length mismatch: actions={actions.shape}, eef_states={eef_states.shape}, "
            f"gripper_states={gripper_states.shape} ({hdf5_path.name}:{demo_key})"
        )
    if eef_states.shape[1] != schema.eef_dim:
        raise ValueError(f"Expected eef_states dim {schema.eef_dim}, got {eef_states.shape}")
    if gripper_states.shape[1] != schema.gripper_dim:
        raise ValueError(f"Expected gripper_states dim {schema.gripper_dim}, got {gripper_states.shape}")
    if actions.shape[1] != schema.action_dim:
        raise ValueError(f"Expected actions dim {schema.action_dim}, got {actions.shape}")

    # Canonical concatenated state: [eef(6), gripper(G)] = state_dim
    obs_state = np.concatenate([eef_states, gripper_states], axis=1).astype(np.float64)  # (T, state_dim)
    action_vec = actions.astype(np.float64)  # (T, action_dim)

    # Split action into "eef" part and "gripper" part (heuristic: last dim is gripper)
    eef_action = action_vec[:, :schema.eef_action_dim] if schema.eef_action_dim > 0 else action_vec[:, :0]
    gripper_cmd = action_vec[:, schema.eef_action_dim:] if schema.gripper_action_dim > 0 else action_vec[:, :0]

    # Per-frame task index: match DROID style (task_index mirrors the primary language annotation)
    lang_col = np.full(T, task_idx, dtype=np.int64)

    episode_dict = {
        "observation.state": obs_state,
        "action": action_vec,
        "annotation.language.language_instruction": lang_col,
        "task_index": lang_col.copy(),
        "timestamp": (np.arange(T, dtype=np.float64) / float(fps)),
        "episode_index": np.full(T, ep_idx, dtype=np.int64),
        "frame_index": np.arange(T, dtype=np.int64),
    }

    if write_debug_columns:
        episode_dict.update(
            {
                "state.joint_position": eef_states.astype(np.float64),          # (T, eef_dim)
                "state.gripper_position": gripper_states.astype(np.float64),    # (T, gripper_dim)
                "action.joint_position": eef_action.astype(np.float64),         # (T, eef_action_dim)
                "action.gripper_position": gripper_cmd.astype(np.float64),      # (T, gripper_action_dim)
            }
        )

    parquet_path = output_path / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
    pl.DataFrame(episode_dict).write_parquet(parquet_path)

    # Videos
    encode_video(
        agentview,
        output_path / f"videos/chunk-{chunk_idx:03d}/observation.images.agentview_rgb/episode_{ep_idx:06d}.mp4",
        fps,
    )
    encode_video(
        eye_in_hand,
        output_path / f"videos/chunk-{chunk_idx:03d}/observation.images.eye_in_hand_rgb/episode_{ep_idx:06d}.mp4",
        fps,
    )

    return {"episode_index": ep_idx, "tasks": [task_name], "length": T}


def _process_episode_worker(args):
    return process_episode(*args)


def convert_libero_split(
    input_dir: str,
    output_dir: str,
    fps: int = 20,
    max_workers: int = 8,
    write_debug_columns: bool = False,
    resolution: int = 128,
    task_suite_name: str = None,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)

    hdf5_files = sorted(input_path.glob("*_demo.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No *_demo.hdf5 files found in {input_dir}")
    print(f"Found {len(hdf5_files)} task files in {input_dir}")

    # Infer schema once (assumes consistent schema across split)
    schema = infer_libero_schema(hdf5_files[0])
    print(
        f"Inferred schema from {hdf5_files[0].name}: "
        f"uses_robot0_keys={schema.uses_robot0_keys}, "
        f"state_dim={schema.obs_state_dim} (eef={schema.eef_dim}, grip={schema.gripper_dim}), "
        f"action_dim={schema.action_dim} (eef_action={schema.eef_action_dim}, grip_action={schema.gripper_action_dim}), "
        f"agentview_src={schema.agentview_src_key}, eye_src={schema.eye_in_hand_src_key}"
    )

    if resolution != 128:
        print(f"Re-rendering at {resolution}x{resolution} (single-threaded, MuJoCo envs cannot be forked)")

    # Build task registry and episode plan
    all_tasks: Dict[str, int] = {}
    task_counter = 0
    episode_plan: List[Tuple] = []
    ep_idx = 0

    for hdf5_file in hdf5_files:
        task_name = hdf5_file.stem.replace("_demo", "")
        if task_name not in all_tasks:
            all_tasks[task_name] = task_counter
            task_counter += 1
        task_idx = all_tasks[task_name]

        bddl_file = None
        if resolution != 128:
            bddl_file = _get_bddl_file(hdf5_file, task_suite_name)

        with h5py.File(hdf5_file, "r") as f:
            demo_keys = sorted(list(f["data"].keys()))

        for demo_key in demo_keys:
            episode_plan.append((ep_idx, hdf5_file, demo_key, task_name, task_idx, bddl_file))
            ep_idx += 1

    total_episodes = len(episode_plan)
    print(f"Total episodes: {total_episodes}, tasks: {len(all_tasks)}")

    # tasks.jsonl
    with open(output_path / "meta/tasks.jsonl", "w") as f:
        for task_name, task_idx in sorted(all_tasks.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": task_idx, "task": task_name}) + "\n")

    # Convert episodes
    worker_args = [
        (
            ep_i,
            hdf5_path,
            demo_key,
            task_name,
            task_idx,
            output_path,
            fps,
            write_debug_columns,
            schema,
            resolution,
            bddl_file,
        )
        for ep_i, hdf5_path, demo_key, task_name, task_idx, bddl_file in episode_plan
    ]

    episodes_data: List[Dict] = []
    if resolution != 128:
        # Re-rendering must be single-threaded (MuJoCo)
        for args in tqdm.tqdm(worker_args, desc="Re-rendering"):
            episodes_data.append(_process_episode_worker(args))
    elif max_workers > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(mp_context=ctx, max_workers=max_workers) as executor:
            futures = [executor.submit(_process_episode_worker, args) for args in worker_args]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Converting"):
                episodes_data.append(future.result())
    else:
        for args in tqdm.tqdm(worker_args, desc="Converting"):
            episodes_data.append(_process_episode_worker(args))

    episodes_data.sort(key=lambda x: x["episode_index"])

    # episodes.jsonl
    with open(output_path / "meta/episodes.jsonl", "w") as f:
        for ep in episodes_data:
            f.write(json.dumps(ep) + "\n")

    # modality.json (DROID-style)
    # state: [eef(6)] as "joint_position", gripper(G) as "gripper_position"
    # action: [eef_action(A-1)] as "joint_position", gripper(1) as "gripper_position"
    state_keys = ["joint_position", "gripper_position"]
    state_dims = [schema.eef_dim, schema.gripper_dim]
    action_keys = ["joint_position", "gripper_position"]
    action_dims = [schema.eef_action_dim, schema.gripper_action_dim]

    modality_config = {
        "state": _concat_info(state_keys, state_dims),
        "action": _concat_info(action_keys, action_dims),
        "video": {k: {"original_key": f"observation.images.{k}"} for k in OUTPUT_IMAGE_KEYS},
        "annotation": {f"language.{lang_key}": {} for lang_key in LANG_KEYS},
    }
    with open(output_path / "meta/modality.json", "w") as f:
        json.dump(modality_config, f, indent=4)

    # info.json
    total_frames = sum(int(ep["length"]) for ep in episodes_data)
    num_chunks = (total_episodes // CHUNKS_SIZE) + (1 if total_episodes % CHUNKS_SIZE else 0)

    info_features = {
        **{
            f"observation.images.{k}": {
                "dtype": "video",
                "shape": [resolution, resolution, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
            for k in OUTPUT_IMAGE_KEYS
        },
        "observation.state": {"dtype": "float64", "shape": [schema.obs_state_dim], "names": state_keys},
        "action": {"dtype": "float64", "shape": [schema.action_dim], "names": action_keys},
        "timestamp": {"dtype": "float64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        **{f"annotation.language.{k}": {"dtype": "int64", "shape": [1]} for k in LANG_KEYS},
    }

    if write_debug_columns:
        info_features.update(
            {
                "state.joint_position": {"dtype": "float64", "shape": [schema.eef_dim], "names": ["joint_position"]},
                "state.gripper_position": {
                    "dtype": "float64",
                    "shape": [schema.gripper_dim],
                    "names": ["gripper_position"],
                },
                "action.joint_position": {
                    "dtype": "float64",
                    "shape": [schema.eef_action_dim],
                    "names": ["joint_position"],
                },
                "action.gripper_position": {
                    "dtype": "float64",
                    "shape": [schema.gripper_action_dim],
                    "names": ["gripper_position"],
                },
            }
        )

    info = {
        "codebase_version": "v2.0",
        "robot_type": "libero",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(all_tasks),
        "total_videos": len(OUTPUT_IMAGE_KEYS),
        "total_chunks": num_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": info_features,
    }

    with open(output_path / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"Done. {total_episodes} episodes, {total_frames} frames → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LIBERO HDF5 demos to LeRobot v2.0 (VLA-compatible).")
    parser.add_argument("input_dir", help="Directory containing *_demo.hdf5 files (one LIBERO split)")
    parser.add_argument("output_dir", help="Output directory for LeRobot dataset")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second (default: 20)")
    parser.add_argument("-n", type=int, default=8, help="Max parallel workers for 128px mode (default: 8)")
    parser.add_argument(
        "--write-debug-columns",
        action="store_true",
        help="Also write decomposed state.* and action.* columns (optional).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Camera render resolution (default: 128). Values > 128 trigger environment replay "
             "to re-render images at the target resolution (single-threaded).",
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default=None,
        help="LIBERO task suite name (e.g. libero_10, libero_spatial). Required when --resolution > 128 "
             "unless the HDF5 files store env_args with the BDDL file path.",
    )
    args = parser.parse_args()

    if args.resolution != 128 and args.task_suite_name is None:
        print("Warning: --task_suite_name not provided; will attempt to read BDDL path from HDF5 env_args.")

    convert_libero_split(
        args.input_dir,
        args.output_dir,
        fps=args.fps,
        max_workers=args.n,
        write_debug_columns=args.write_debug_columns,
        resolution=args.resolution,
        task_suite_name=args.task_suite_name,
    )