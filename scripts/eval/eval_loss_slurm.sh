#!/bin/bash
#SBATCH --job-name=dreamzero_eval_loss
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=0-1:00:00
#SBATCH --output=logs/eval_loss_%j.out
#SBATCH --error=logs/eval_loss_%j.err
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END,FAIL

# Usage:
#   sbatch scripts/eval/eval_loss_slurm.sh
# Override checkpoint:
#   CHECKPOINT=.../checkpoint-1000 sbatch scripts/eval/eval_loss_slurm.sh

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/.bashrc
conda deactivate
conda activate vla

export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ============ USER CONFIGURATION ============
CHECKPOINT=${CHECKPOINT:-"/n/netscratch/sham_lab/Lab/chloe00/libero/dreamzero_libero_all_lora/checkpoint-2800"}
NUM_BATCHES=${NUM_BATCHES:-20}
# =============================================

mkdir -p logs

torchrun --standalone --nproc_per_node=2 scripts/eval_loss_on_data.py \
    --checkpoint "$CHECKPOINT" \
    --num-batches "$NUM_BATCHES"
