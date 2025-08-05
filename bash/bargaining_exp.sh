#!/bin/bash
#SBATCH --job-name=bargaining_exp
#SBATCH --output=/gpfs/radev/project/zhuoran_yang/cl2637/exp_updated/joboutput/%x_%j.out
#SBATCH --error=/gpfs/radev/project/zhuoran_yang/cl2637/exp_updated/joboutput/%x_%j.err
#SBATCH --partition=day         
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00          # 10 hours

module load miniconda

# initialise Conda for non‑interactive shell and activate env
source activate base
conda activate py310

set -euo pipefail

echo
echo "=============================================================="
echo "  Starting bargaining job …"
echo "=============================================================="

# ── 2) Change to your project root
PROJECT_DIR="/gpfs/radev/project/zhuoran_yang/cl2637/exp_updated/COOPA"      # <-- edit if needed
cd "$PROJECT_DIR"

# ── 3) User-configurable parameters
data_split="validation"
random_seed=30
num_workers=15
log_dir="logs"
mode="uniform"

# Optional: restrict which models participate
# Example → model_filter="gpt Qwen"
model_filter=""

# ── 4) Ensure log directory exists
mkdir -p "$log_dir"

# ── 5) Build optional --model_filter argument
model_filter_arg=""
if [[ -n "$model_filter" ]]; then
  model_filter_arg="--model_filter $model_filter"
fi

# ── 6) Launch Python (single run, handles all pairs)
python -m apps.bargaining.run \
       --data_split "$data_split" \
       --mode "$mode" \
       --random_seed "$random_seed" \
       --num_workers "$num_workers" \
       --log_dir "$log_dir" \
       $model_filter_arg

echo
echo "All experiments completed!"