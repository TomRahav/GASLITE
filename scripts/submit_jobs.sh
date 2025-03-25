#!/bin/bash

# Define arrays of parameters 
models=("Snowflake/snowflake-arctic-embed-m" "sentence-transformers/all-MiniLM-L6-v2" "intfloat/e5-base-v2") # "Snowflake/snowflake-arctic-embed-m" "sentence-transformers/all-MiniLM-L6-v2" "intfloat/e5-base-v2"
concepts=("potter" "flower" "vaccine" "boston" "golf" "iphone" "mortgage" "sandwich") #  "potter" "flower" "vaccine" "boston" "golf" "iphone" "mortgage" "sandwich"
methods=("reguler") # "reguler" "monte_carlo" "avg_loss" "avg_loss_onehot_grad" "avg_loss_weighted_linearly"
mal_info_lengths=("short" "medium" "long") # "short" "medium" "long"
trigger_lens=(10 20 30) # 10 20 30
attack_n_iters=(100) # 30 100 300

index=0  # Initialize counter

# Iterate over each combination of parameters
for concept in "${concepts[@]}"; do
  for method in "${methods[@]}"; do
    for attack_n_iter in "${attack_n_iters[@]}"; do
      for trigger_len in "${trigger_lens[@]}"; do
        for mal_info_length in "${mal_info_lengths[@]}"; do
          for model in "${models[@]}"; do

            # Submit the job with the current set of parameters
            sbatch --export=CONCEPT="$concept",MAL_INFO_LENGTH="$mal_info_length",METHOD="$method",TRIGGER_LEN="$trigger_len",ATTACK_N_ITER="$attack_n_iter",MODEL="$model",RANDOM_SEED="$index" scripts/job_script.sh

            ((index++))  # Increment counter
            sleep 1  # Brief pause to be kind to the scheduler

          done
        done
      done
    done
  done
done