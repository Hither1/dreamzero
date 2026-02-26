#!/bin/bash
# DreamZero LIBERO Finetuning Script
#
# Usage:
#   # Convert data first (one-time):
#   python scripts/data/convert_libero.py \
#       /n/netscratch/sham_lab/Lab/chloe00/libero/libero/datasets/libero_spatial \
#       ./data/libero_spatial_lerobot
#
#   # Then run finetuning:
#   LIBERO_DATA_ROOT=./data/libero_spatial_lerobot bash scripts/train/libero_training.sh
#
# Prerequisites:
#   - LIBERO dataset converted to LeRobot format at LIBERO_DATA_ROOT
#     (run scripts/data/convert_libero.py first)
#   - Pretrained DreamZero checkpoint or Wan2.1-I2V-14B-480P base weights
#     Download: huggingface-cli download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir ./checkpoints/dreamzero_droid
#   - umt5-xxl tokenizer
#     Download: huggingface-cli download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl

export HYDRA_FULL_ERROR=1

# ============ USER CONFIGURATION ============
# LeRobot-format LIBERO dataset (output of convert_libero.py)
LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-"./data/libero_spatial_lerobot"}

# Output directory for finetuned checkpoints
OUTPUT_DIR=${OUTPUT_DIR:-"/n/netscratch/sham_lab/Lab/chloe00/dreamzero_libero_lora"}

# Number of GPUs
NUM_GPUS=${NUM_GPUS:-4}

# Pretrained DreamZero checkpoint (recommended) or Wan2.1 base weights
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"/n/netscratch/sham_lab/Lab/chloe00/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/n/netscratch/sham_lab/Lab/chloe00/umt5-xxl"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
# ================================================

# Validate dataset exists
if [ ! -d "$LIBERO_DATA_ROOT" ]; then
    echo "ERROR: LIBERO dataset not found at $LIBERO_DATA_ROOT"
    echo "Convert it first with:"
    echo "  python scripts/data/convert_libero.py <input_hdf5_dir> $LIBERO_DATA_ROOT"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/libero_relative \
    wandb_project=dreamzero_libero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=200 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=1000 \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    libero_data_root=$LIBERO_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
