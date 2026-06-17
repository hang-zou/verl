#!/usr/bin/env bash
# Customized verl GRPO FSDP recipe for TelecomLLM.
# Wraps examples/grpo_trainer/run_qwen3_4b_fsdp.sh with:
#   - local Qwen3-4B-Instruct-2507 checkpoint
#   - local GSM8K parquets
#   - wandb logging (project: verl-test)
#   - SMOKE_TEST toggle (defaults to 1 -> 5 steps, no save, no eval)
#
# Usage:
#   bash run_qwen3_4b_grpo.sh                          # smoke test (default)
#   SMOKE_TEST=0 bash run_qwen3_4b_grpo.sh             # full 15-epoch run
#   EXP_NAME=my_run bash run_qwen3_4b_grpo.sh          # custom wandb run name
#   bash run_qwen3_4b_grpo.sh actor_rollout_ref.actor.optim.lr=5e-7   # extra Hydra overrides

set -e
source /apps/ku/intel_h200_gpu/miniconda/3/etc/profile.d/conda.sh
conda activate verl

# ---- local paths (override-friendly) ----
export MODEL_PATH=${MODEL_PATH:-/dpc/kuin0100/hang/Documents/models/Qwen/Qwen3-4B-Instruct-2507}
export TRAIN_FILE=${TRAIN_FILE:-/dpc/kuin0100/hang/Documents/datasets/hf/openai/gsm8k/train.parquet}
export TEST_FILE=${TEST_FILE:-/dpc/kuin0100/hang/Documents/datasets/hf/openai/gsm8k/test.parquet}

# ---- wandb (uses ~/.netrc auth; project/experiment set via Hydra below) ----
unset WANDB_MODE   # in case a previous shell set WANDB_MODE=disabled

# ---- smoke-test vs full ----
SMOKE_TEST=${SMOKE_TEST:-1}
EXP_NAME=${EXP_NAME:-qwen3_4b_grpo_$(date +%Y%m%d_%H%M%S)}

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
  EXTRA_OVERRIDES=(
    trainer.save_freq=20
    trainer.test_freq=5
  )
fi

# ---- log ----
mkdir -p run_logs
LOG=run_logs/${EXP_NAME}.log
echo "log: $LOG"
echo "wandb run: verl-test/${EXP_NAME}"
echo "smoke_test: ${SMOKE_TEST}"

# ---- launch ----
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  trainer.project_name=verl-test \
  trainer.experiment_name="$EXP_NAME" \
  trainer.default_local_dir=/dpc/kuin0100/hang/Documents/checkpoints/verl-test/$EXP_NAME \
  "${EXTRA_OVERRIDES[@]}" \
  "$@" \
  2>&1 | tee "$LOG"
