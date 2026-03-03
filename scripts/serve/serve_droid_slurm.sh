#!/bin/bash
#SBATCH --job-name=dreamzero_droid_serve
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/dreamzero_droid_serve_%j.out
#SBATCH --error=logs/dreamzero_droid_serve_%j.err

# DreamZero DROID Real-Robot Policy Server
#
# Starts the DreamZero WebSocket server on a cluster H100/A100 node.
# The DROID robot workstation connects to this server via droid_client.py.
#
# Usage:
#   sbatch scripts/serve/serve_droid_slurm.sh
#
# After the job starts, find the server address with:
#   grep "server\|hostname\|ip\|port" logs/dreamzero_droid_serve_<jobid>.out
#
# Then on the robot workstation:
#   python droid_client.py --host <hostname> --port 8000 --prompt "..."
#
# To use A100 instead of H100, change the partition to:
#   #SBATCH --partition=kempner

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

source ~/.bashrc
conda activate vla

# ============ USER CONFIGURATION ============
# Path to your DreamZero DROID checkpoint.
# Must be the checkpoint root (the directory that contains experiment_cfg/).
# Hardcoded here so a stale MODEL_PATH env var from ~/.bashrc cannot override it.
MODEL_PATH="/n/netscratch/sham_lab/Lab/chloe00/libero/checkpoints/DreamZero-DROID"

# WebSocket port (must be reachable from the robot workstation)
PORT=${PORT:-8000}

# Number of GPUs — auto-detected from the SLURM allocation so it always
# matches --gres=gpu:N and --ntasks-per-node=N above.
# Override with: NUM_GPUS=4 sbatch ...
NUM_GPUS=${NUM_GPUS:-${SLURM_NTASKS_PER_NODE:-8}}

# Index suffix for output directory naming (useful when running multiple servers)
SERVER_INDEX=${SERVER_INDEX:-0}

# Optional: enable DiT KV cache for faster inference (experimental)
# Set to "true" to enable: ENABLE_DIT_CACHE=true sbatch ...
ENABLE_DIT_CACHE=${ENABLE_DIT_CACHE:-false}
# =============================================

mkdir -p logs

# Print the hostname so the robot workstation knows where to connect
echo "============================================"
echo "Server hostname: $(hostname)"
echo "Server IP:       $(hostname -I | awk '{print $1}')"
echo "Server port:     $PORT"
echo "Model path:      $MODEL_PATH"
echo "GPUs:            $NUM_GPUS"
echo "Job ID:          $SLURM_JOB_ID"
echo "============================================"

export PYTHONPATH=$PYTHONPATH:$PWD

# Redirect HuggingFace cache away from home dir (which has limited quota)
# to netscratch so model components don't try to download into ~/.cache.
export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HUB_CACHE"

# tyro parses bool fields as flags (--flag / omit), not --flag true/false.
# Build the optional --enable_dit_cache flag only when requested.
DIT_CACHE_FLAG=""
if [ "$ENABLE_DIT_CACHE" = "true" ]; then
    DIT_CACHE_FLAG="--enable_dit_cache"
fi

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --standalone \
    socket_test_optimized_AR.py \
    --port $PORT \
    --model_path "$MODEL_PATH" \
    $DIT_CACHE_FLAG \
    --index $SERVER_INDEX
