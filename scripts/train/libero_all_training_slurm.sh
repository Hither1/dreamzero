#!/bin/bash
#SBATCH --job-name=dreamzero_libero_all
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=1-6:00:00
#SBATCH --output=logs/dreamzero_libero_all_%j.out
#SBATCH --error=logs/dreamzero_libero_all_%j.err
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END

# DreamZero LIBERO Finetuning Script (all 4 suites combined)
#
# Trains on libero_spatial + libero_goal + libero_object + libero_10
# Datasets must already be converted to LeRobot format (they are at ./data/*_lerobot)
#
# Usage:
#   sbatch scripts/train/libero_all_training_slurm.sh

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800  # 3h heartbeat — eval can block ranks
# ddp_timeout is passed to dist.init_process_group(timeout=...) via HF Trainer.
# Default is 1800s (30 min); LIBERO eval on rank 0 can take >30 min with many tasks.
# Set to 3h to prevent NCCL operation timeout during inline eval.

source ~/.bashrc
conda deactivate
conda activate vla

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export LIBERO_CONFIG_PATH=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/third_party/libero
export PYTHONPATH=$PYTHONPATH:/n/netscratch/sham_lab/Lab/chloe00/libero
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero
export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ============ USER CONFIGURATION ============
OUTPUT_DIR=${OUTPUT_DIR:-"/n/netscratch/sham_lab/Lab/chloe00/libero/dreamzero_libero_all_lora"}
NUM_GPUS=${NUM_GPUS:-4}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"/n/netscratch/sham_lab/Lab/chloe00/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/n/netscratch/sham_lab/Lab/chloe00/umt5-xxl"}
# =============================================

mkdir -p logs

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

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/libero_all_relative \
    wandb_project=dreamzero_libero_all \
    train_architecture=lora \
    num_frames=9 \
    action_horizon=48 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=48 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=200 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=4 \
    max_steps=82000 \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    gradient_checkpointing=true \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=224 \
    image_resolution_height=224 \
    save_lora_only=false \
    libero_eval_task_suite=libero_10 \
    libero_eval_num_trials=3 \
    libero_eval_max_tasks=5 \
    libero_eval_on_save=false \
    ++libero_eval_on_train_begin=false \
    libero_eval_max_steps_per_episode=200 \
    ++training_args.ddp_timeout=10800 \
    max_chunk_size=1 \
    frame_seqlen=784 \
    save_strategy=steps \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=/n/netscratch/sham_lab/Lab/chloe00/libero/checkpoints/DreamZero-DROID \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
