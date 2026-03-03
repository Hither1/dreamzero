# DreamZero — DROID Real Robot Server/Client

This guide explains how to run DreamZero inference against a physical DROID robot
by hosting the model on a cluster (H100/A100) and connecting from the robot workstation over WebSocket.

```
[DROID Robot Workstation]  ──── WebSocket ────  [Cluster H100/A100]
     droid_client.py                             socket_test_optimized_AR.py
     - Camera capture                            - DreamZero 14B inference
     - Joint state reading                       - torchrun multi-GPU
     - Action execution                          - Port 8000
```

---

## Files

| File | Location | Purpose |
|---|---|---|
| `socket_test_optimized_AR.py` | repo root | **Server** — distributed WebSocket inference server |
| `droid_client.py` | repo root | **Client** — DROID robot observation/action adapter |
| `scripts/serve/serve_droid_slurm.sh` | scripts/ | SLURM job script to launch the server on the cluster |
| `eval_utils/policy_server.py` | eval_utils/ | WebSocket server base (roboarena interface) |
| `eval_utils/policy_client.py` | eval_utils/ | WebSocket client base |
| `test_client_AR.py` | repo root | Smoke-test client using dummy or real video frames |

---

## Step 1 — Start the server on the cluster

### With SLURM (recommended)

```bash
# From the dreamzero/ directory on the cluster:
sbatch scripts/serve/serve_droid_slurm.sh
```

After the job starts, find the server hostname in the SLURM `.out` log:

```bash
grep "hostname\|IP\|port" logs/dreamzero_droid_serve_<JOBID>.out
# Server hostname: holygpu8a17102.rc.fas.harvard.edu
# Server IP:       10.31.146.142
# Server port:     8000
```

The server is ready when you see:

```
INFO:websockets.server:server listening on 0.0.0.0:8000
```

### Directly with torchrun (interactive session)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --standalone --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port 8000 \
    --model_path /path/to/DreamZero-DROID
```

### SLURM configuration

Edit `scripts/serve/serve_droid_slurm.sh` to change:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `...DreamZero-DROID` | Checkpoint root (must contain `experiment_cfg/`) |
| `PORT` | `8000` | WebSocket port |
| `ENABLE_DIT_CACHE` | `false` | Set `true` for ~5× faster inference (GB200 only) |
| `--ntasks-per-node` / `--gres=gpu:N` | `2` | Must match each other; use 8 for full H100 node |

> **Important:** `MODEL_PATH` must be the checkpoint root directory (the folder that contains
> `experiment_cfg/conf.yaml`), **not** the `experiment_cfg/` subdirectory itself.

---

## Step 2 — Quick smoke test (no robot needed)

Run `test_client_AR.py` from any machine that can reach the cluster node:

```bash
conda activate vla
cd dreamzero/

python test_client_AR.py \
    --host holygpu8a17102.rc.fas.harvard.edu \
    --port 8000 \
    --use-zero-images \
    --num-chunks 2 \
    --prompt "pick up the red cup"
```

Expected output per inference call:

```
Action shape: (24, 8), range: [-0.0123, 0.0456], time: 3.21s
```

The first 1-2 calls will be slower due to PyTorch compilation warmup.
Subsequent calls should be ~3s on H100, ~0.6s on GB200 with `--enable_dit_cache`.

For a more realistic test with real video frames, place MP4 files in `debug_image/`
(see `test_client_AR.py` docstring) and omit `--use-zero-images`.

---

## Step 3 — Connect the real DROID robot

`droid_client.py` is a template that you adapt to your specific DROID setup.

### 1. Implement the robot interface

Open `droid_client.py` and fill in the three `NotImplementedError` stubs in `DROIDRobotEnv`:

```python
class DROIDRobotEnv:
    def __init__(self):
        # Example using droid.robot_env.RobotEnv:
        from droid.robot_env import RobotEnv
        self.env = RobotEnv(action_space="joint_position", ...)

    def get_observation(self) -> dict:
        # Returns: right_image, left_image, wrist_image (H,W,3 uint8)
        #          joint_position (7,), gripper_position (1,)
        raw = self.env.get_observation()
        return {
            "right_image":       raw["camera_obs"]["ext1"]["array"],
            "left_image":        raw["camera_obs"]["ext2"]["array"],
            "wrist_image":       raw["camera_obs"]["wrist"]["array"],
            "joint_position":    raw["robot_state"]["joint_positions"].astype(np.float64),
            "gripper_position":  np.array([raw["robot_state"]["gripper_position"]], dtype=np.float64),
        }

    def step(self, joint_positions: np.ndarray, gripper: float) -> None:
        action = np.concatenate([joint_positions, [gripper]])
        self.env.step(action)

    def reset(self) -> None:
        self.env.reset()
```

### 2. Run the client

```bash
# On the robot workstation:
conda activate vla
cd dreamzero/

python droid_client.py \
    --host holygpu8a17102.rc.fas.harvard.edu \
    --port 8000 \
    --prompt "pick up the red cup" \
    --episodes 5 \
    --max-steps 300 \
    --open-loop-horizon 8
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--host` | *(required)* | Cluster node hostname from the SLURM log |
| `--port` | `8000` | Must match server `--port` |
| `--prompt` | *(required)* | Language instruction |
| `--episodes` | `1` | Number of consecutive episodes |
| `--max-steps` | `300` | Max timesteps per episode |
| `--open-loop-horizon` | `8` | Actions to execute before querying server again |

### Observation / action format

The client sends per inference call:

| Key | Shape | Description |
|---|---|---|
| `observation/exterior_image_0_left` | `(180, 320, 3)` uint8 | Right external camera |
| `observation/exterior_image_1_left` | `(180, 320, 3)` uint8 | Left external camera |
| `observation/wrist_image_left` | `(180, 320, 3)` uint8 | Wrist camera |
| `observation/joint_position` | `(7,)` float64 | Arm joint angles (rad) |
| `observation/cartesian_position` | `(6,)` float64 | Dummy zeros acceptable |
| `observation/gripper_position` | `(1,)` float64 | Gripper width [0, 1] |
| `prompt` | str | Language instruction |
| `session_id` | str | UUID, resets server state on change |

The server returns `(N, 8)` float32 — N action steps of 7 joint positions + 1 gripper.
The client executes these `N` actions open-loop before querying again.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Unrecognized arguments: false` | `tyro` treats bools as flags | Use `--enable_dit_cache` (no value), never `--enable_dit_cache false` |
| `torch._dynamo.config.recompile_limit does not exist` | Renamed in PyTorch 2.6 | Use `cache_size_limit` — already patched in `socket_test_optimized_AR.py` |
| `CUDA error: invalid device ordinal` | `--nproc_per_node` > actual GPUs | `--ntasks-per-node` in SLURM must equal `--gres=gpu:N`; script auto-detects via `$SLURM_NTASKS_PER_NODE` |
| `FileNotFoundError: experiment_cfg/conf.yaml` | Wrong `MODEL_PATH` | Pass the checkpoint root (parent of `experiment_cfg/`), not the subdirectory |
| `OSError: No space left on device` (HuggingFace cache) | `~/.cache` home quota full | Script sets `HF_HOME` to netscratch — pre-download Wan2.1 there before first run |
| Connection refused | Server still loading | Wait for `server listening on 0.0.0.0:8000` in the `.out` log |
