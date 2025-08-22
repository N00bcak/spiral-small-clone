# Copyright 2025 SPIRAL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Common =========
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0
export LP_DEBUG=1
export LP_LOG_LEVEL=DEBUG
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7,8 # GPU 4 is the 4090.
# export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO  # helps with debugging

# Abstracted batch sizes
GPUS=8
ROLLOUT_BATCH_SIZE_PER_DEVICE=16
TRAIN_BATCH_SIZE_PER_DEVICE=1
ROLLOUT_BATCH_SIZE=$((GPUS * ROLLOUT_BATCH_SIZE_PER_DEVICE))
TRAIN_BATCH_SIZE=$((GPUS * ROLLOUT_BATCH_SIZE_PER_DEVICE))

# Notes ==========
# Setting `--save_steps 16` to save checkpoints every 16 policy iteration steps.
# Set `--eval_opponent_names google/gemini-2.0-flash-lite-001` if you have OpenRouter access.
# The public repo does NOT support training on multiple game environments.
# Ensure rollout_batch_size = rollout_batch_size_per_device * gpus.
# Beta is the KL Divergence coefficient, set to 0 for no KL penalty.
# Max length should be set quite generously; it's rare for models to generate that long.
python train_spiral.py \
    --env_ids KuhnPoker-v1 \
    --use_llm_obs_wrappers True \
    --eval_env_ids KuhnPoker-v1 \
    --eval_use_llm_obs_wrappers True \
    --eval_opponent_names HF:spiral-rl/Spiral-Qwen3-4B \
    --eval_split all \
    --gamma 1 \
    --gpus $GPUS \
    --gradient-checkpointing \
    --num_samples 1 \
    --num_envs 1 \
    --pretrain Qwen/Qwen3-0.6B-Base \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.3 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 2 \
    --gradient-checkpointing \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --rollout_batch_size_per_device $ROLLOUT_BATCH_SIZE_PER_DEVICE \
    --pi_buffer_maxlen_per_device $ROLLOUT_BATCH_SIZE_PER_DEVICE \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --dump_game_state_every 1 \
    --train_batch_size_per_device 1 \
    --beta 0.001 \
    --max_model_len 12800 \
    --generate_max_length 4096 \
    --max_context_length 32768 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps 32 \
    --save_steps 32 \
    --eval_games 8 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 3072 \
    --max_train 25600 \
    --save_path ~/nfs/spiral-small-clone/spiral-small-self-play-pilot \
    --max_save_num 8 \
    --use-wb \
    --wb-run-name spiral-small-self-play-pilot \
    --wb_project spiral-small-clone