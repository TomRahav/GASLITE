#!/bin/bash
#SBATCH --job-name=GASLITE_Attack
#SBATCH --output=logs/output_%j.txt    # Standard output
#SBATCH --error=logs/error_%j.txt      # Standard error
#SBATCH --time=01:00:00                # Time limit hrs:min:sec
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --gres=gpu:A40:1               # Request 1 GPU of type A40
#SBATCH --mail-user=tom.rahav@campus.technion.ac.il
#SBATCH --mail-type=ALL                # Send email on all events

# Load necessary modules
# module purge
# module load matlab/R2023a

# Activate conda environment
source activate gaslite

# Set constant parameters
DATASET=msmarco-train-concepts
MODEL=Snowflake/snowflake-arctic-embed-m
SIM_FUNC=cos_sim
RANDOM_SEED=0
BATCH_SIZE=2048

# Run the Python script with the parameters passed from the environment
python hydra_entrypoint.py --config-name default \
  model.sim_func_name=${SIM_FUNC} \
  model.model_hf_name=${MODEL} \
  dataset=${DATASET} \
  core_objective=single-query \
  batch_size=${BATCH_SIZE} \
  random_seed=${RANDOM_SEED} \
  exp_tag=exp0_knows-all \
  cover_alg=concept-test-${CONCEPT} \
  ++constraints.trigger_len=${TRIGGER_LEN} \
  ++mal_info_length=${MAL_INFO_LENGTH} \
  ++chunk_robustness_method=${METHOD} \
  ++attack.attack_n_iter=30 \
  attack.beam_search_config.n_flip=500 \
  ++test_chunking=end
