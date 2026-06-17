#!/usr/bin/env bash
# Megatron-LM sibling of run_qwen2_5_7b_grpo.sh — same model, same dataset (GSM8K),
# same wandb project, same H200 tuning. Wraps the upstream Megatron recipe
# examples/grpo_trainer/run_qwen2-7b_math_megatron_fsdp.sh and overrides its small
# default batches/lengths so it is APPLES-TO-APPLES with the FSDP launcher.
#
# Requires the local _telecomllm_patches.py monkey-patch (auto-loaded via
# verl/__init__.py) — without it, Megatron's _set_attention_backend asserts when
# verl builds the actor and ref with different attention backends.
#
# Usage:
#   bash run_qwen2_5_7b_grpo_megatron.sh                          # smoke (5 steps)
#   SMOKE_TEST=0 bash run_qwen2_5_7b_grpo_megatron.sh             # full 15-epoch run
#   EXP_NAME=my_run bash run_qwen2_5_7b_grpo_megatron.sh          # custom wandb run name
#   bash run_qwen2_5_7b_grpo_megatron.sh actor_rollout_ref.actor.optim.lr=5e-7   # extra overrides

set -e
source /apps/ku/intel_h200_gpu/miniconda/3/etc/profile.d/conda.sh
conda activate verl

# ---- local paths ----
export HF_MODEL_PATH=${HF_MODEL_PATH:-/dpc/kuin0100/hang/Documents/models/Qwen/Qwen2.5-7B-Instruct}
GSM8K_TRAIN=/dpc/kuin0100/hang/Documents/datasets/hf/openai/gsm8k/train.parquet
GSM8K_TEST=/dpc/kuin0100/hang/Documents/datasets/hf/openai/gsm8k/test.parquet
export gsm8k_train_path=$GSM8K_TRAIN
export gsm8k_test_path=$GSM8K_TEST
# Force the recipe's `train_files`/`test_files` to GSM8K only (its default is GSM8K+MATH).
export train_files="['$GSM8K_TRAIN']"
export test_files="['$GSM8K_TEST']"

# ---- wandb ----
unset WANDB_MODE   # in case a previous shell set WANDB_MODE=disabled

# ---- Megatron parallelism (recipe envs) ----
export TP=${TP:-4}        # actor training tensor parallel
export PP=${PP:-1}        # actor training pipeline parallel
export GEN_TP=${GEN_TP:-2}  # rollout (vLLM) TP — match FSDP launcher's ROLLOUT_TP=2

# ---- NVTE attention env vars (defensive; patch handles cross-build conflicts now) ----
export NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN:-1}
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1}
export NVTE_UNFUSED_ATTN=${NVTE_UNFUSED_ATTN:-1}

# ---- smoke vs full ----
SMOKE_TEST=${SMOKE_TEST:-1}
EXP_NAME=${EXP_NAME:-qwen2_5_7b_grpo_megatron_$(date +%Y%m%d_%H%M%S)}

if [ "$SMOKE_TEST" = "1" ]; then
  TRAIN_BATCH_SIZE=32
  PPO_MINI_BATCH_SIZE=16
  ROLLOUT_N=2
  EXTRA_OVERRIDES=(
    trainer.total_training_steps=5
    trainer.total_epochs=1
    trainer.save_freq=1000
    trainer.test_freq=1000
  )
else
  TRAIN_BATCH_SIZE=512
  PPO_MINI_BATCH_SIZE=128
  ROLLOUT_N=8
  # NOTE: save_freq deliberately set above total_training_steps. Megatron-core 0.13.1
  # + mbridge distrib_optimizer raises "Model param ... not in model_sharded_state_dict"
  # at save time. Skip saves until that's resolved. See [[feedback-megatron-save-bug]].
  EXTRA_OVERRIDES=(
    trainer.save_freq=99999
    trainer.test_freq=5
  )
fi

# ---- Apples-to-apples knobs vs run_qwen2_5_7b_grpo.sh (FSDP) ----
APPLES=(
  # data shape — recipe default is 512/512; mirror FSDP launcher (1024/2048)
  data.train_batch_size=$TRAIN_BATCH_SIZE
  data.max_prompt_length=1024
  data.max_response_length=2048
  # actor PPO config — use dynamic batching like the FSDP path
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
  # NOTE: actor_rollout_ref.actor.entropy_from_logits_with_chunking is FSDP-only;
  # Megatron's actor schema doesn't register it and Hydra rejects it. Removed.
  # rollout
  actor_rollout_ref.rollout.n=$ROLLOUT_N
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.max_num_batched_tokens=8192
  # ref
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24000
  # trainer
  trainer.balance_batch=True
  # bridge — use ISEEKYAN/mbridge path even though the patch makes it backend-agnostic
  actor_rollout_ref.actor.megatron.vanilla_mbridge=True
  actor_rollout_ref.ref.megatron.vanilla_mbridge=True
)

# ---- log ----
mkdir -p run_logs
LOG=run_logs/${EXP_NAME}.log
echo "log: $LOG"
echo "wandb run: verl-test/${EXP_NAME}"
echo "smoke_test: ${SMOKE_TEST}"
echo "model: ${HF_MODEL_PATH}"
echo "parallelism: TP=$TP PP=$PP GEN_TP=$GEN_TP"

# ---- launch ----
bash examples/grpo_trainer/run_qwen2-7b_math_megatron_fsdp.sh \
  trainer.project_name=verl-test \
  trainer.experiment_name="$EXP_NAME" \
  trainer.default_local_dir=/dpc/kuin0100/hang/Documents/checkpoints/verl-test/$EXP_NAME \
  "${APPLES[@]}" \
  "${EXTRA_OVERRIDES[@]}" \
  "$@" \
  2>&1 | tee "$LOG"
