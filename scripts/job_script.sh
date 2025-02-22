#!/bin/bash
#SBATCH --job-name=GASLITE_Attack
#SBATCH --output=logs/job_%j.txt    # Standard output
#SBATCH --error=logs/job_%j.txt      # Standard error
#SBATCH --time=01:00:00                # Time limit hrs:min:sec
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH -c 8                      # number of cores (treats)
#SBATCH --gres=gpu:A40:1          # Request 1 gpu type A40
#SBATCH --mail-user=tom.rahav@campus.technion.ac.il
#SBATCH --mail-type=NONE                # Send email on all events

# Load necessary modules
source /home/tom.rahav/miniconda3/etc/profile.d/conda.sh
export WAND_API_KEY=a8edcfc5767b7efdd613b561c389b6951eee54e7
# Activate conda environment
conda activate gaslite

# Set constant parameters
DATASET=msmarco-train-concepts
MODEL=Snowflake/snowflake-arctic-embed-m # Snowflake/snowflake-arctic-embed-m | sentence-transformers/all-MiniLM-L6-v2
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
  cover_alg=concept-test-${CONCEPT} \
  ++constraints.trigger_len=${TRIGGER_LEN} \
  ++mal_info_length=${MAL_INFO_LENGTH} \
  ++chunk_robustness_method=${METHOD} \
  ++attack.attack_n_iter=${ATTACK_N_ITER} \
  attack.beam_search_config.n_flip=500 \
  ++test_chunking=end \
  exp_tag="[exp-test, malinfo-${MAL_INFO_LENGTH}, robustness-${METHOD}, concept-${CONCEPT}, triggerlen-${TRIGGER_LEN}, attack_n_iter-${ATTACK_N_ITER}]"

