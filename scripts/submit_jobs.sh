#!/bin/bash

# Define arrays of parameters
concepts=("potter" "flower" "vaccine")
methods=("regular" "monte_carlo" "avg_loss")
mal_info_lengths=("short" "medium" "long")
trigger_lens=(10 20 30)

# Iterate over each combination of parameters
for concept in "${concepts[@]}"; do
  for mal_info_length in "${mal_info_lengths[@]}"; do
    for method in "${methods[@]}"; do
      for trigger_len in "${trigger_lens[@]}"; do
        # Submit the job with the current set of parameters
        sbatch --export=CONCEPT="$concept",MAL_INFO_LENGTH="$mal_info_length",METHOD="$method",TRIGGER_LEN="$trigger_len" job_script.sh
        sleep 1 # Brief pause to be kind to the scheduler
      done
    done
  done
done
