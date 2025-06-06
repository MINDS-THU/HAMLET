#!/bin/bash

# Define the list of models
models=("gpt-4.1" "claude-sonnet-4-20250514" "gemini-2.5-pro" "Qwen/Qwen3-32B" "google/gemma-3-27b-it" "mistralai/Mistral-Small-3.1-24B-Instruct-2503")

# Define the data splits to evaluate
data_splits=("validation")

# Define the random seed for reproducibility
random_seed=30

# Define whether to use gain from trade
gain_from_trade=true

# Create a directory for logs
log_dir="./logs"
mkdir -p "$log_dir"

# Create a temporary file to store commands
commands_file="./commands.txt"
rm -f "$commands_file"

# Generate commands for all combinations of buyer and seller models
for buyer_model in "${models[@]}"; do
    for seller_model in "${models[@]}"; do
        for data_split in "${data_splits[@]}"; do
            echo "python ./apps/bargaining/run.py \
                --buyer_model \"$buyer_model\" \
                --seller_model \"$seller_model\" \
                --data_split \"$data_split\" \
                --gain_from_trade \"$gain_from_trade\" \
                --random_seed \"$random_seed\" \
                > \"$log_dir/${buyer_model}_${seller_model}_${data_split}.log\" 2>&1" >> "$commands_file"
        done
    done
done

# Prompt the user for the number of parallel processes
read -p "Enter the number of parallel processes: " num_parallel

# Run the commands in parallel
parallel -j "$num_parallel" < "$commands_file"

# Clean up the temporary file
rm -f "$commands_file"

echo "All experiments completed!"