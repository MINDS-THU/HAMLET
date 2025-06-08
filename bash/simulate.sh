#!/bin/bash

echo "Starting bargaining simulation batch job..."

# Loop from data_idx 0 to 929
for data_idx in {320..500}; do
  echo "Running simulation for data_idx=$data_idx with buyer=1.2, seller=0.8"
  python -m apps.bargaining.simulate \
    --data_idx "$data_idx" \
    --buyer_fraction 1.2 \
    --seller_fraction 0.8

  echo "Running simulation for data_idx=$data_idx with buyer=0.8, seller=1.2"
  python -m apps.bargaining.simulate \
    --data_idx "$data_idx" \
    --buyer_fraction 0.8 \
    --seller_fraction 1.2
done

echo "Simulation batch job completed."

