"""
Convert LIBERO demonstration datasets (HDF5) to LeRobot v2.0 format
in a way that aligns with DreamZero/VLA finetuning expectations.

This script mirrors the key schema conventions used by convert_droid.py:
- observation.state : concatenated float64 vector (T, 8) = [joint_states(7), gripper(1)]
- action            : concatenated float64 vector (T, 7) = LIBERO actions (typically 6 eef delta + 1 gripper)
- annotation.language.language_instruction : int64 per-frame task index
- task_index        : int64 per-frame task index (copied from the language annotation)
- timestamp, episode_index, frame_index

It also optionally emits decomposed "debug" columns:
- state.joint_position, state.gripper_position, action.joint_position

Output structure:
  <output_dir>/
    meta/modality.json
    meta/episodes.jsonl
    meta/tasks.jsonl
    meta/info.json
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.agentview_rgb/episode_000000.mp4
    videos/chunk-000/observation.images.eye_in_hand_rgb/episode_000000.mp4

Usage:
  python scripts/train/convert_libero.py \
      /path/to/libero_split_dir \
      /path/to/output_dir \
      --fps 20 -n 8
"""

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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


IMAGE_KEYS = ["agentview_rgb", "eye_in_hand_rgb"]
LANG_KEYS = ["language_instruction"]
CHUNKS_SIZE = 1000

JOINT_DIM = 7
GRIPPER_DIM = 1
OBS_STATE_DIM = JOINT_DIM + GRIPPER_DIM  # 8
ACTION_DIM = 7  # LIBERO actions are typically (6 eef delta + 1 gripper)


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


def _read_demo_arrays(
    hdf5_path: Path, demo_key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      actions: (T, 7)
      joint_states: (T, 7)
      gripper_states: (T, 1)
      agentview: (T, 128, 128, 3)
      eye_in_hand: (T, 128, 128, 3)
    """
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_key]

        actions = demo["actions"][:]  # (T, 7)

        joint_states = demo["obs"]["joint_states"][:]  # (T, 7)

        gripper_raw = demo["obs"]["gripper_states"][:]  # often (T,2) or (T,1)
        if gripper_raw.ndim == 2 and gripper_raw.shape[1] >= 1:
            gripper_states = gripper_raw[:, :1]
        else:
            gripper_states = gripper_raw.reshape(-1, 1)

        agentview = demo["obs"]["agentview_rgb"][:]      # (T, 128, 128, 3)
        eye_in_hand = demo["obs"]["eye_in_hand_rgb"][:]  # (T, 128, 128, 3)

    return actions, joint_states, gripper_states, agentview, eye_in_hand


def _concat_info(keys: List[str], dims: List[int]) -> Dict[str, Dict[str, int]]:
    """Build concat_info like convert_droid.py uses for modality.json."""
    assert len(keys) == len(dims)
    out: Dict[str, Dict[str, int]] = {}
    start = 0
    for k, d in zip(keys, dims):
        out[k] = {"start": start, "end": start + int(d)}
        start += int(d)
    return out


def process_episode(
    ep_idx: int,
    hdf5_path: Path,
    demo_key: str,
    task_name: str,
    task_idx: int,
    output_path: Path,
    fps: int,
    write_debug_columns: bool,
) -> Dict:
    """Convert one LIBERO demo to parquet + videos. Returns episode metadata dict."""
    chunk_idx = ep_idx // CHUNKS_SIZE

    (output_path / f"data/chunk-{chunk_idx:03d}").mkdir(parents=True, exist_ok=True)
    for img_key in IMAGE_KEYS:
        (output_path / f"videos/chunk-{chunk_idx:03d}/observation.images.{img_key}").mkdir(
            parents=True, exist_ok=True
        )

    actions, joint_states, gripper_states, agentview, eye_in_hand = _read_demo_arrays(hdf5_path, demo_key)

    T = int(actions.shape[0])
    if joint_states.shape[0] != T or gripper_states.shape[0] != T:
        raise ValueError(
            f"Length mismatch: actions={actions.shape}, joint_states={joint_states.shape}, gripper_states={gripper_states.shape} "
            f"({hdf5_path.name}:{demo_key})"
        )
    if joint_states.shape[1] != JOINT_DIM:
        raise ValueError(f"Expected joint_states dim {JOINT_DIM}, got {joint_states.shape}")
    if gripper_states.shape[1] != GRIPPER_DIM:
        raise ValueError(f"Expected gripper_states dim {GRIPPER_DIM}, got {gripper_states.shape}")
    if actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions dim {ACTION_DIM}, got {actions.shape}")

    # Canonical concatenated vectors
    obs_state = np.concatenate([joint_states, gripper_states], axis=1).astype(np.float64)  # (T, 8)
    action_vec = actions.astype(np.float64)  # (T, 7)

    # Per-frame task index: match DROID style (task_index mirrors the primary language annotation)
    lang_col = np.full(T, task_idx, dtype=np.int64)

    episode_dict = {
        "observation.state": obs_state,
        "action": action_vec,
        "annotation.language.language_instruction": lang_col,
        "task_index": lang_col.copy(),

        "timestamp": (np.arange(T, dtype=np.float64) / float(fps)).reshape(-1),
        "episode_index": np.full(T, ep_idx, dtype=np.int64).reshape(-1),
        "frame_index": np.arange(T, dtype=np.int64).reshape(-1),
    }

    # Optional debug columns (won't hurt unless your loader is strict about unknown columns)
    if write_debug_columns:
        episode_dict["state.joint_position"] = joint_states.astype(np.float64)
        episode_dict["state.gripper_position"] = gripper_states.astype(np.float64)
        episode_dict["action.joint_position"] = actions.astype(np.float64)

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
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)

    hdf5_files = sorted(input_path.glob("*_demo.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No *_demo.hdf5 files found in {input_dir}")
    print(f"Found {len(hdf5_files)} task files in {input_dir}")

    # Build task registry and episode plan
    all_tasks: Dict[str, int] = {}
    task_counter = 0
    episode_plan: List[Tuple[int, Path, str, str, int]] = []
    ep_idx = 0

    for hdf5_file in hdf5_files:
        task_name = hdf5_file.stem.replace("_demo", "")
        if task_name not in all_tasks:
            all_tasks[task_name] = task_counter
            task_counter += 1
        task_idx = all_tasks[task_name]

        with h5py.File(hdf5_file, "r") as f:
            demo_keys = sorted(list(f["data"].keys()))

        for demo_key in demo_keys:
            episode_plan.append((ep_idx, hdf5_file, demo_key, task_name, task_idx))
            ep_idx += 1

    total_episodes = len(episode_plan)
    print(f"Total episodes: {total_episodes}, tasks: {len(all_tasks)}")

    # tasks.jsonl
    with open(output_path / "meta/tasks.jsonl", "w") as f:
        for task_name, task_idx in sorted(all_tasks.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": task_idx, "task": task_name}) + "\n")

    # Convert episodes
    worker_args = [
        (ep_i, hdf5_path, demo_key, task_name, task_idx, output_path, fps, write_debug_columns)
        for ep_i, hdf5_path, demo_key, task_name, task_idx in episode_plan
    ]

    episodes_data: List[Dict] = []
    if max_workers > 1:
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
    # Here, we provide concat_info for state and action so downstream code can reason about sub-keys.
    # For LIBERO:
    # - state keys: joint_position(7), gripper_position(1) => observation.state dim 8
    # - action keys: joint_position(7) (name kept to match your earlier convention)
    state_keys = ["joint_position", "gripper_position"]
    state_dims = [JOINT_DIM, GRIPPER_DIM]
    action_keys = ["joint_position"]
    action_dims = [ACTION_DIM]

    modality_config = {
        "state": _concat_info(state_keys, state_dims),
        "action": _concat_info(action_keys, action_dims),
        "video": {k: {"original_key": f"observation.images.{k}"} for k in IMAGE_KEYS},
        "annotation": {f"language.{lang_key}": {} for lang_key in LANG_KEYS},
    }
    with open(output_path / "meta/modality.json", "w") as f:
        json.dump(modality_config, f, indent=4)

    # info.json
    total_frames = sum(int(ep["length"]) for ep in episodes_data)
    num_chunks = (total_episodes // CHUNKS_SIZE) + (1 if total_episodes % CHUNKS_SIZE else 0)

    info_features = {
        # Video features
        **{
            f"observation.images.{k}": {
                "dtype": "video",
                "shape": [128, 128, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
            for k in IMAGE_KEYS
        },
        # Canonical concatenated features
        "observation.state": {"dtype": "float64", "shape": [OBS_STATE_DIM], "names": state_keys},
        "action": {"dtype": "float64", "shape": [ACTION_DIM], "names": action_keys},

        # Scalars
        "timestamp": {"dtype": "float64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},

        # Language annotation
        **{f"annotation.language.{k}": {"dtype": "int64", "shape": [1]} for k in LANG_KEYS},
    }

    if write_debug_columns:
        info_features.update(
            {
                "state.joint_position": {"dtype": "float64", "shape": [JOINT_DIM], "names": ["joint_position"]},
                "state.gripper_position": {"dtype": "float64", "shape": [GRIPPER_DIM], "names": ["gripper_position"]},
                "action.joint_position": {"dtype": "float64", "shape": [ACTION_DIM], "names": ["joint_position"]},
            }
        )

    info = {
        "codebase_version": "v2.0",
        "robot_type": "libero",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(all_tasks),
        "total_videos": len(IMAGE_KEYS),
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
    parser.add_argument("-n", type=int, default=8, help="Max parallel workers (default: 8)")
    parser.add_argument(
        "--write-debug-columns",
        action="store_true",
        help="Also write decomposed state.* and action.joint_position columns (optional).",
    )
    args = parser.parse_args()

    convert_libero_split(
        args.input_dir,
        args.output_dir,
        fps=args.fps,
        max_workers=args.n,
        write_debug_columns=args.write_debug_columns,
    )