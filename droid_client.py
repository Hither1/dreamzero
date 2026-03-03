"""DROID Real Robot Client for DreamZero policy server on cluster.

Runs on the robot workstation. Connects to the DreamZero model
server (H100/A100) via WebSocket and executes predicted actions.

Architecture:
  [DROID Robot Workstation]  ──WebSocket──  [Cluster H100/A100]
       droid_client.py                      socket_test_optimized_AR.py
       - Camera capture                     - DreamZero inference
       - Joint state reading                - torchrun multi-GPU
       - Action execution                   - WebSocket server on port 8000

Observation format sent to server:
  observation/exterior_image_0_left:  (H, W, 3) uint8  right external camera
  observation/exterior_image_1_left:  (H, W, 3) uint8  left external camera
  observation/wrist_image_left:       (H, W, 3) uint8  wrist camera
  observation/joint_position:         (7,) float64     arm joint angles (rad)
  observation/cartesian_position:     (6,) float64     dummy zeros acceptable
  observation/gripper_position:       (1,) float64     gripper width [0, 1]
  prompt:                             str              language instruction
  session_id:                         str              unique episode ID

Server returns:
  action: (N, 8) float32  N steps × (7 joint targets + 1 gripper)

Usage:
  # 1. Start server on cluster (see scripts/serve/serve_droid_slurm.sh):
  torchrun --nproc_per_node=8 socket_test_optimized_AR.py \\
      --port 8000 --model_path /path/to/dreamzero/checkpoint

  # 2. Run on DROID robot workstation:
  python droid_client.py \\
      --host <cluster_node_hostname_or_ip> \\
      --port 8000 \\
      --prompt "pick up the red cup" \\
      --episodes 5
"""

import argparse
import logging
import time
import uuid

import numpy as np

from eval_utils.policy_client import WebsocketClientPolicy

logger = logging.getLogger(__name__)

# Server expects images at 180×320 (H×W) — matches DROID training resolution
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 320

# Number of actions to execute before querying the server again (open-loop chunk)
DEFAULT_OPEN_LOOP_HORIZON = 8


# ---------------------------------------------------------------------------
# Robot interface — fill in with your DROID library calls
# ---------------------------------------------------------------------------

class DROIDRobotEnv:
    """Adapter for the DROID physical robot.

    Replace each stub with the actual DROID library call for your setup.
    The typical DROID Python interface is ``droid.robot_env.RobotEnv``.

    Example init::

        from droid.robot_env import RobotEnv
        self.env = RobotEnv(action_space="joint_position",
                            camera_kwargs={"image_hw": (480, 640)})
    """

    def __init__(self):
        # TODO: initialize your DROID robot environment here
        raise NotImplementedError(
            "Implement DROIDRobotEnv.__init__ with your robot setup.\n"
            "Example:\n"
            "  from droid.robot_env import RobotEnv\n"
            "  self.env = RobotEnv(action_space='joint_position', ...)"
        )

    def get_observation(self) -> dict:
        """Read the current robot observation.

        Returns a dict with keys:
          right_image:       (H, W, 3) uint8   right/primary external camera
          left_image:        (H, W, 3) uint8   left/secondary external camera
          wrist_image:       (H, W, 3) uint8   wrist camera
          joint_position:    (7,) float64      arm joint angles in radians
          gripper_position:  (1,) float64      gripper opening [0=closed, 1=open]
        """
        # TODO: replace with actual calls
        # Example with droid.robot_env.RobotEnv:
        #   raw = self.env.get_observation()
        #   cam = raw["camera_obs"]
        #   state = raw["robot_state"]
        #   return {
        #       "right_image": cam["ext1"]["array"],           # (H, W, 3) uint8
        #       "left_image":  cam["ext2"]["array"],           # (H, W, 3) uint8
        #       "wrist_image": cam["wrist"]["array"],          # (H, W, 3) uint8
        #       "joint_position":   state["joint_positions"].astype(np.float64),  # (7,)
        #       "gripper_position": np.array([state["gripper_position"]], dtype=np.float64),  # (1,)
        #   }
        raise NotImplementedError("Implement DROIDRobotEnv.get_observation()")

    def step(self, joint_positions: np.ndarray, gripper: float) -> None:
        """Send a joint-position command to the robot.

        Args:
            joint_positions: (7,) target joint angles in radians
            gripper:         0.0 = fully closed, 1.0 = fully open
        """
        # TODO: replace with actual calls
        # Example with droid.robot_env.RobotEnv:
        #   action = np.concatenate([joint_positions, [gripper]])
        #   self.env.step(action)
        raise NotImplementedError("Implement DROIDRobotEnv.step()")

    def reset(self) -> None:
        """Move robot to home position between episodes."""
        # TODO: replace with actual calls
        # Example: self.env.reset()
        raise NotImplementedError("Implement DROIDRobotEnv.reset()")


# ---------------------------------------------------------------------------
# Observation / action helpers
# ---------------------------------------------------------------------------

def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize image to (height, width) with letterbox padding.

    Falls back to a simple cv2 resize when openpi_client is unavailable.
    """
    try:
        from openpi_client import image_tools
        return image_tools.resize_with_pad(image, height, width)
    except ImportError:
        import cv2  # type: ignore
        resized = cv2.resize(image, (width, height))
        return resized


def build_server_obs(raw_obs: dict, prompt: str, session_id: str) -> dict:
    """Convert raw DROID observation to the server's expected dict format."""
    return {
        "observation/exterior_image_0_left": _resize_with_pad(
            raw_obs["right_image"], IMAGE_HEIGHT, IMAGE_WIDTH
        ),
        "observation/exterior_image_1_left": _resize_with_pad(
            raw_obs["left_image"], IMAGE_HEIGHT, IMAGE_WIDTH
        ),
        "observation/wrist_image_left": _resize_with_pad(
            raw_obs["wrist_image"], IMAGE_HEIGHT, IMAGE_WIDTH
        ),
        "observation/joint_position":    raw_obs["joint_position"].astype(np.float64),
        "observation/cartesian_position": np.zeros(6, dtype=np.float64),
        "observation/gripper_position":  raw_obs["gripper_position"].astype(np.float64),
        "prompt":     prompt,
        "session_id": session_id,
    }


def binarize_gripper(value: float) -> float:
    """Threshold gripper command: >0.5 → open (1.0), else → closed (0.0)."""
    return 1.0 if value > 0.5 else 0.0


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def run_episode(
    robot: DROIDRobotEnv,
    client: WebsocketClientPolicy,
    prompt: str,
    session_id: str,
    max_steps: int = 300,
    open_loop_horizon: int = DEFAULT_OPEN_LOOP_HORIZON,
) -> None:
    """Run one policy episode on the real DROID robot.

    The robot queries the server every ``open_loop_horizon`` steps and
    executes the returned action chunk open-loop in between.
    """
    action_chunk: np.ndarray | None = None
    actions_executed = 0

    for step in range(max_steps):
        # Re-query the server when the current chunk is exhausted
        if action_chunk is None or actions_executed >= open_loop_horizon:
            raw_obs = robot.get_observation()
            server_obs = build_server_obs(raw_obs, prompt, session_id)

            logger.info("Step %d: querying server...", step)
            t0 = time.perf_counter()
            action_chunk = client.infer(server_obs)
            dt = time.perf_counter() - t0

            assert action_chunk.ndim == 2 and action_chunk.shape[-1] == 8, (
                f"Expected (N, 8) action array, got {action_chunk.shape}"
            )
            logger.info(
                "  chunk shape=%s  range=[%.3f, %.3f]  latency=%.2fs",
                action_chunk.shape,
                float(action_chunk.min()),
                float(action_chunk.max()),
                dt,
            )
            actions_executed = 0

        # Execute the next action from the cached chunk
        action = action_chunk[actions_executed]
        joint_pos = action[:7]
        gripper = binarize_gripper(float(action[7]))
        robot.step(joint_pos, gripper)
        actions_executed += 1

    logger.info("Episode finished after %d steps.", max_steps)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DROID real-robot client for DreamZero cluster server"
    )
    parser.add_argument(
        "--host", required=True,
        help="Hostname or IP of the cluster node running socket_test_optimized_AR.py",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--prompt", required=True, help="Language instruction for the robot")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument(
        "--open-loop-horizon", type=int, default=DEFAULT_OPEN_LOOP_HORIZON,
        help=f"Actions to execute per server query (default: {DEFAULT_OPEN_LOOP_HORIZON})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Connecting to DreamZero server at %s:%d", args.host, args.port)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    logger.info("Server metadata: %s", client.get_server_metadata())

    robot = DROIDRobotEnv()

    for ep in range(args.episodes):
        session_id = str(uuid.uuid4())
        logger.info(
            "=== Episode %d/%d | session=%s ===", ep + 1, args.episodes, session_id
        )
        logger.info("Prompt: '%s'", args.prompt)

        run_episode(
            robot=robot,
            client=client,
            prompt=args.prompt,
            session_id=session_id,
            max_steps=args.max_steps,
            open_loop_horizon=args.open_loop_horizon,
        )

        # Signal server to reset its internal state (frame buffers, KV cache, etc.)
        client.reset({})
        robot.reset()

    logger.info("All episodes complete.")


if __name__ == "__main__":
    main()
