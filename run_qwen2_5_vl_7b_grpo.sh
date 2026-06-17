#!/usr/bin/env bash
# Customized verl GRPO FSDP launcher for the Qwen2.5-VL-7B multimodal recipe.
# Wraps examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh with:
#   - local Qwen2.5-VL-7B-Instruct checkpoint
#   - local Geo3K parquets (hiyouga/geometry3k)
#   - wandb logging (project: verl-test)
#   - SMOKE_TEST toggle (defaults to 1 -> 5 steps, no save, no eval)
#
# Usage:
#   bash run_qwen2_5_vl_7b_grpo.sh                          # smoke (5 steps)
#   SMOKE_TEST=0 bash run_qwen2_5_vl_7b_grpo.sh             # full 15-epoch run
#   EXP_NAME=my_run bash run_qwen2_5_vl_7b_grpo.sh          # custom wandb run name
#   bash run_qwen2_5_vl_7b_grpo.sh actor_rollout_ref.actor.optim.lr=5e-7   # extra Hydra overrides

set -e
source /apps/ku/intel_h200_gpu/miniconda/3/etc/profile.d/conda.sh
conda activate verl

# ---- local paths ----
export MODEL_PATH=${MODEL_PATH:-/dpc/kuin0100/hang/Documents/models/Qwen/Qwen2.5-VL-7B-Instruct}
TRAIN_FILE=/dpc/kuin0100/hang/Documents/datasets/hf/hiyouga/geometry3k/train.parquet
TEST_FILE=/dpc/kuin0100/hang/Documents/datasets/hf/hiyouga/geometry3k/test.parquet

# ---- wandb ----
unset WANDB_MODE   # in case a previous shell set WANDB_MODE=disabled

# ---- smoke vs full ----
SMOKE_TEST=${SMOKE_TEST:-1}
EXP_NAME=${EXP_NAME:-qwen2_5_vl_7b_grpo_$(date +%Y%m%d_%H%M%S)}

if [ "$SMOKE_TEST" = "1" ]; then
  export TRAIN_BATCH_SIZE=32
  export PPO_MINI_BATCH_SIZE=16
  export ROLLOUT_N=2
  EXTRA_OVERRIDES=(
    trainer.total_training_steps=5
    trainer.total_epochs=1
    trainer.save_freq=1000
    trainer.test_freq=1000
  )
else
  # Recipe defaults are already H200-tuned: batch=512, mini=128, rollout_n=5,
  # max_response=2048, ppo_max_token_len_per_gpu=24576.
  EXTRA_OVERRIDES=(
    trainer.save_freq=20
    trainer.test_freq=5
  )
fi

# Override Geo3K paths (recipe defaults to $HOME/data/geo3k/*).
DATA_OVERRIDES=(
  "data.train_files=$TRAIN_FILE"
  "data.val_files=$TEST_FILE"
)

# MLLM FSDP2 fix: the recipe's default use_fused_kernels=True crashes the first
# log_prob compute with a mixed torch.Tensor / DTensor matmul in
# verl/utils/experimental/torch_functional.py. Disable until that path is
# DTensor-aware. See [[feedback-mllm-fused-kernels-off]].
MLLM_FIX=(
  actor_rollout_ref.model.use_fused_kernels=False
)

# ---- log ----
mkdir -p run_logs
LOG=run_logs/${EXP_NAME}.log
echo "log: $LOG"
echo "wandb run: verl-test/${EXP_NAME}"
echo "smoke_test: ${SMOKE_TEST}"
echo "model: ${MODEL_PATH}"
echo "data: $TRAIN_FILE"

# ---- launch ----
bash examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh \
  trainer.project_name=verl-test \
  trainer.experiment_name="$EXP_NAME" \
  trainer.default_local_dir=/dpc/kuin0100/hang/Documents/checkpoints/verl-test/$EXP_NAME \
  "${DATA_OVERRIDES[@]}" \
  "${MLLM_FIX[@]}" \
  "${EXTRA_OVERRIDES[@]}" \
  "$@" \
  2>&1 | tee "$LOG"
