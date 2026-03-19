"""Compute training loss on LIBERO finetuning data using a DreamZero checkpoint.

Loads the model and data pipeline from a saved checkpoint directory and runs
the exact same forward pass used during training (model(batch) → outputs["loss"]).

Requires 2 GPUs (tensor parallelism) via torchrun.

Usage:
    cd /path/to/vla-interp/dreamzero
    torchrun --standalone --nproc_per_node=2 scripts/eval_loss_on_data.py \
        --checkpoint /n/netscratch/sham_lab/Lab/chloe00/libero/dreamzero_libero_all_lora/checkpoint-2800 \
        --num-batches 20
"""

import argparse
import datetime
import pathlib
import pickle
import sys

import json

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from omegaconf import OmegaConf
from hydra.utils import instantiate
from safetensors.torch import load_file

_DREAMZERO_DIR = str(pathlib.Path(__file__).resolve().parents[1])
if _DREAMZERO_DIR not in sys.path:
    sys.path.insert(0, _DREAMZERO_DIR)

from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig


def load_model(checkpoint_dir: pathlib.Path) -> VLA:
    """Instantiate VLA with PEFT and load weights without the buggy .base_layer. stripping.

    VLA.from_pretrained strips '.base_layer.' from checkpoint keys before loading.
    This is wrong when the checkpoint was saved with PEFT active (save_lora_only=False):
    both the checkpoint AND the freshly-instantiated PEFT model use '.base_layer.' for
    base weights, so stripping it makes those keys 'unexpected' and the base weights
    are silently dropped.  We bypass that by loading the state dict directly.
    """
    # Read config and disable defer_lora_injection so PEFT is injected during __init__
    config_path = checkpoint_dir / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)
    config = VLAConfig(**config_dict)
    if isinstance(config.action_head_cfg.get("config"), dict):
        config.action_head_cfg["config"]["defer_lora_injection"] = False

    # Instantiate model (LoRA is injected immediately, model keys now include .base_layer.)
    model = VLA(config)

    # Load sharded safetensors WITHOUT any key manipulation
    index_path = checkpoint_dir / "model.safetensors.index.json"
    single_path = checkpoint_dir / "model.safetensors"
    state_dict = {}
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        for shard_file in set(index["weight_map"].values()):
            state_dict.update(load_file(str(checkpoint_dir / shard_file)))
    elif single_path.exists():
        state_dict = load_file(str(single_path))
    else:
        raise FileNotFoundError(f"No safetensors found in {checkpoint_dir}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return model


def move_to_device(batch: dict, device: str) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def load_config(checkpoint_dir: pathlib.Path) -> OmegaConf:
    candidates = [
        checkpoint_dir / "experiment_cfg" / "conf.yaml",
        checkpoint_dir.parent / "experiment_cfg" / "conf.yaml",
    ]
    for p in candidates:
        if p.exists():
            return OmegaConf.load(p)
    raise FileNotFoundError(
        f"Could not find experiment_cfg/conf.yaml relative to {checkpoint_dir}"
    )


def _broadcast_batch(batch: dict) -> dict:
    """Rank 0 sends a batch dict to all other ranks."""
    data = pickle.dumps(batch)
    size = torch.tensor([len(data)], dtype=torch.int64, device="cuda")
    dist.broadcast(size, src=0)
    buf = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    dist.broadcast(buf, src=0)
    return batch


def _receive_batch() -> dict:
    """Non-rank-0 receives a batch dict from rank 0."""
    size = torch.zeros(1, dtype=torch.int64, device="cuda")
    dist.broadcast(size, src=0)
    buf = torch.zeros(int(size.item()), dtype=torch.uint8, device="cuda")
    dist.broadcast(buf, src=0)
    return pickle.loads(buf.cpu().numpy().tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DreamZero training loss on LIBERO finetuning data"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a checkpoint-XXXX subdirectory"
    )
    parser.add_argument(
        "--num-batches", type=int, default=20,
        help="Number of batches to evaluate (default: 20)"
    )
    args = parser.parse_args()

    # ── Distributed init ────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=1))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    checkpoint_dir = pathlib.Path(args.checkpoint)
    assert checkpoint_dir.exists(), f"Checkpoint dir not found: {checkpoint_dir}"

    # ── 1. Load model on all ranks ───────────────────────────────────────────
    if rank == 0:
        print(f"\n=== Loading model from {checkpoint_dir} ===")
    model = load_model(checkpoint_dir)
    # Convert to bf16 on CPU before moving to GPU (~28 GB per model copy,
    # but tensor parallelism splits it across ranks so ~14 GB per GPU).
    model = model.to(torch.bfloat16)

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ip",))
    model.post_initialize()
    model.parallelize(mesh)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    if rank == 0:
        print("Model loaded and parallelized across", world_size, "GPUs.")

    # signal tensor used to tell workers to stop (must be on CUDA for NCCL)
    signal = torch.zeros(1, dtype=torch.int32, device=device)

    # ── 2. Rank 0: load dataset + collator ──────────────────────────────────
    if rank == 0:
        cfg = load_config(checkpoint_dir)
        print("\n=== Instantiating dataset ===")
        dataset = instantiate(cfg.train_dataset)
        collator = instantiate(cfg.data_collator)
        print(f"\n=== Computing loss on {args.num_batches} batches ===")

    # ── 3. Evaluation loop ───────────────────────────────────────────────────
    losses = []
    action_losses = []
    dynamics_losses = []

    if rank == 0:
        dataset_iter = iter(dataset)

    for i in range(args.num_batches):
        if rank == 0:
            try:
                raw_sample = next(dataset_iter)
            except StopIteration:
                # Signal workers to stop early
                signal.fill_(1)
                dist.broadcast(signal, src=0)
                break
            signal.fill_(0)
            dist.broadcast(signal, src=0)

            batch = collator([raw_sample])
            _broadcast_batch(batch)
        else:
            dist.broadcast(signal, src=0)
            if signal.item() == 1:
                break
            batch = _receive_batch()

        batch = move_to_device(batch, device)

        dist.barrier()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(batch)
        dist.barrier()

        if rank == 0:
            loss = outputs["loss"].item()
            losses.append(loss)

            action_loss = outputs.get("action_loss", outputs.get("action_loss_avg"))
            dynamics_loss = outputs.get("dynamics_loss", outputs.get("dynamics_loss_avg"))

            parts = [f"loss={loss:.4f}"]
            if action_loss is not None:
                al = action_loss.item() if isinstance(action_loss, torch.Tensor) else float(action_loss)
                action_losses.append(al)
                parts.append(f"action_loss={al:.4f}")
            if dynamics_loss is not None:
                dl = dynamics_loss.item() if isinstance(dynamics_loss, torch.Tensor) else float(dynamics_loss)
                dynamics_losses.append(dl)
                parts.append(f"dynamics_loss={dl:.4f}")

            print(f"  batch {i:3d}: {', '.join(parts)}")

    # ── 4. Summary (rank 0 only) ─────────────────────────────────────────────
    if rank == 0:
        if not losses:
            print("No batches completed.")
        else:
            print(f"\n{'='*50}")
            print(f"Batches evaluated : {len(losses)}")
            print(f"Loss              : mean={np.mean(losses):.4f}  "
                  f"std={np.std(losses):.4f}  "
                  f"min={np.min(losses):.4f}  max={np.max(losses):.4f}")
            if action_losses:
                print(f"Action loss       : mean={np.mean(action_losses):.4f}  "
                      f"std={np.std(action_losses):.4f}")
            if dynamics_losses:
                print(f"Dynamics loss     : mean={np.mean(dynamics_losses):.4f}  "
                      f"std={np.std(dynamics_losses):.4f}")
            print(f"{'='*50}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
