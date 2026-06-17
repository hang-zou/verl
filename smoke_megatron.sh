#!/usr/bin/env bash
# Megatron GRPO smoke test (Qwen2.5-7B local, GSM8K+MATH, 10 steps).
# Known to fail at NVTE attention-backend assertion (Megatron _set_attention_backend);
# kept for debugging that path. Run from the verl/ directory.
set -e

source /apps/ku/intel_h200_gpu/miniconda/3/etc/profile.d/conda.sh
conda activate verl

mkdir -p run_logs
LOG=run_logs/smoke_qwen2_5_7b_$(date +%Y%m%d_%H%M%S).log
echo "log: $LOG"

export HF_MODEL_PATH=/dpc/kuin0100/hang/Documents/models/Qwen/Qwen2.5-7B-Instruct
export gsm8k_train_path=/dpc/kuin0100/hang/Documents/datasets/hf/openai/gsm8k/train.parquet
export gsm8k_test_path=/dpc/kuin0100/hang/Documents/datasets/hf/openai/gsm8k/test.parquet
export math_train_path=/dpc/kuin0100/hang/Documents/datasets/hf/DigitalLearningGmbH/MATH-lighteval/train.parquet
export math_test_path=/dpc/kuin0100/hang/Documents/datasets/hf/DigitalLearningGmbH/MATH-lighteval/test.parquet
export WANDB_MODE=disabled
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=1
export NVTE_UNFUSED_ATTN=1

bash examples/grpo_trainer/run_qwen2-7b_math_megatron_fsdp.sh \
  trainer.logger='[console]' \
  trainer.project_name=telecomllm_smoke \
  trainer.experiment_name=qwen2_5_7b_grpo_smoke \
  trainer.total_training_steps=10 \
  trainer.total_epochs=1 \
  trainer.save_freq=1000 \
  trainer.test_freq=1000 \
  actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
  actor_rollout_ref.ref.megatron.vanilla_mbridge=True \
  2>&1 | tee "$LOG"
