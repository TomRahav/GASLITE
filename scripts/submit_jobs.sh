#!/bin/bash

# Define arrays of parameters 
concepts=("flower" "vaccine" "boston" "golf" "iphone" "mortgage" "sandwich") #  "potter" "flower" "vaccine" "boston" "golf" "iphone" "mortgage" "sandwich"
methods=("avg_loss") # "reguler" "monte_carlo" "avg_loss"
mal_info_lengths=("short" "medium" "long") # "short" "medium" "long"
trigger_lens=(10 20 30) # 10 20 30
attack_n_iters=(30 100 300) # 30 100 300

# Iterate over each combination of parameters
for concept in "${concepts[@]}"; do
  for mal_info_length in "${mal_info_lengths[@]}"; do
    for method in "${methods[@]}"; do
      for trigger_len in "${trigger_lens[@]}"; do
        for attack_n_iter in "${attack_n_iters[@]}"; do
          # Submit the job with the current set of parameters
          sbatch --export=CONCEPT="$concept",MAL_INFO_LENGTH="$mal_info_length",METHOD="$method",TRIGGER_LEN="$trigger_len",ATTACK_N_ITER="$attack_n_iter" scripts/job_script.sh
          sleep 1 # Brief pause to be kind to the scheduler
        done
      done
    done
  done
done
