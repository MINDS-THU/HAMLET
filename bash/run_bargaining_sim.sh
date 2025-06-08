#!/bin/bash
#SBATCH --job-name=bargaining_exp
#SBATCH --output=/gpfs/radev/project/zhuoran_yang/cl2637/exp/joboutput/%x_%j.out
#SBATCH --error=/gpfs/radev/project/zhuoran_yang/cl2637/exp/joboutput/%x_%j.err
#SBATCH --partition=day         
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00          # 10 hours

module load miniconda

# initialise Conda for non‑interactive shell and activate env
source activate base
conda activate py310

# go to project root so relative paths work
cd /gpfs/radev/project/zhuoran_yang/cl2637/exp/COOPA

# --------------------------- configuration ----------------------------------
models=(
  "gpt-4.1"
  "Qwen/Qwen3-32B"
  "google/gemma-3-27b-it"
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
)

skip_pairs=(
  "gpt-4.1::gpt-4.1"
  "google/gemma-3-27b-it::google/gemma-3-27b-it"
  "gpt-4.1::Qwen/Qwen3-32B"
  "Qwen/Qwen3-32B::Qwen/Qwen3-32B"
  # removed dangling entry "Qwen/Qwen3-32B::"
  "gpt-4.1::google/gemma-3-27b-it"
)

data_splits=("validation")
random_seed=30
gain_from_trade="True"           # literal True / False
num_workers=8

log_dir="./logs"
mkdir -p "$log_dir"

sanitize() { printf '%s' "$1" | tr '/:' '_' ; }

skip_pair() {
  local candidate="$1"
  for p in "${skip_pairs[@]}"; do [[ "$p" == "$candidate" ]] && return 0 ; done
  return 1
}

# --------------------------- experiment loop --------------------------------
for buyer_model in "${models[@]}"; do
  for seller_model in "${models[@]}"; do
    for data_split in "${data_splits[@]}"; do
      echo "▶︎  Buyer=$buyer_model | Seller=$seller_model | Split=$data_split"

      key="${buyer_model}::${seller_model}"
      if skip_pair "$key"; then
        echo "⏩  Skipping $key (already completed)"
        continue
      fi

      log_file="$log_dir/$(sanitize "$buyer_model")_$(sanitize "$seller_model")_${data_split}.log"

      cmd=(python -m apps.bargaining.run
           --buyer_model "$buyer_model"
           --seller_model "$seller_model"
           --data_split   "$data_split"
           --random_seed  "$random_seed"
           --num_workers  "$num_workers"
           --gain_from_trade "$gain_from_trade")

      # use srun if your site enforces it; otherwise plain exec is fine
      "${cmd[@]}" > "$log_file" 2>&1

      echo "✓  Finished (log: $log_file)"
    done
  done
done

echo "🏁  All experiments completed!"
