#! /bin/sh

# Attacking a single query, by crafting an adversarial passage with GASLITE
# (in paper this is repeated for 50 random queries, using different seeds)

DATASET=msmarco-train-concepts
MODEL=Snowflake/snowflake-arctic-embed-m #sentence-transformers/all-MiniLM-L6-v2
SIM_FUNC=cos_sim
RANDOM_SEED=0  # determines the sampled query
BATCH_SIZE=2048
COVER_ALG=kmeans-test

# Define an array of concepts
concepts=("potter"  "flower")
methods=("reguler" "monte_carlo" "avg_loss")
mal_info_lengths=("short" "medium" "long")
trigger_lens=(10 20 30)

# Run the attack
# (set model and dataset -> set attack parameters (to attack a single query) -> additional config)
# Iterate over each concept
for concept in "${concepts[@]}"; do
  for mal_info_length in "${mal_info_lengths[@]}"; do
    for method in "${methods[@]}"; do
      for trigger_len in "${trigger_lens[@]}"; do
        python hydra_entrypoint.py --config-name default model.sim_func_name=${SIM_FUNC} \
          "model.model_hf_name=${MODEL}" dataset=${DATASET} core_objective=single-query \
          batch_size=${BATCH_SIZE} "random_seed=${RANDOM_SEED}" exp_tag=exp0_knows-all \
          "cover_alg=concept-test-${concept}" ++constraints.trigger_len=${trigger_len} \
          ++mal_info_length=${mal_info_length} ++chunk_robustness_method=${method}
      done
    done
  done
done