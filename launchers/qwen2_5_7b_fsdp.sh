#!/usr/bin/env bash
# Customized verl GRPO FSDP recipe for TelecomLLM, tuned for 8x H200 141GB.
# Wraps examples/grpo_trainer/run_qwen3_4b_fsdp.sh (same FSDP path) with:
#   - local Qwen2.5-7B-Instruct checkpoint
#   - local GSM8K parquets
#   - wandb logging (project: verl-test)
#   - SMOKE_TEST toggle (defaults to 1 -> 5 steps, no save, no eval)
#   - H200-tuned token budgets, rollout parallelism, FSDP knobs
#     (sources: verl perf_tuning docs + examples/tuning/lora/run_qwen3_8b_fsdp.sh)
#
# Sibling to qwen3_4b_fsdp.sh so the two base models can be compared on the same dataset/algorithm.
#
# Usage (run from anywhere — script cd's to verl/ root):
#   bash launchers/qwen2_5_7b_fsdp.sh                          # smoke test (default)
#   SMOKE_TEST=0 bash launchers/qwen2_5_7b_fsdp.sh             # full 15-epoch run
#   EXP_NAME=my_run bash launchers/qwen2_5_7b_fsdp.sh          # custom wandb run name
#   bash launchers/qwen2_5_7b_fsdp.sh actor_rollout_ref.actor.optim.lr=5e-7   # extra Hydra overrides

set -e
source /apps/ku/intel_h200_gpu/miniconda/3/etc/profile.d/conda.sh
conda activate verl

# Run from verl/ root regardless of invocation cwd.
cd "$(dirname "$(readlink -f "$0")")/.."

# ---- local paths (override-friendly) ----
export MODEL_PATH=${MODEL_PATH:-/dpc/kuin0100/hang/Documents/models/Qwen/Qwen2.5-7B-Instruct}
export TRAIN_FILE=${TRAIN_FILE:-/dpc/kuin0100/hang/Documents/datasets/hf/openai/gsm8k/train.parquet}
export TEST_FILE=${TEST_FILE:-/dpc/kuin0100/hang/Documents/datasets/hf/openai/gsm8k/test.parquet}

# ---- wandb (uses ~/.netrc auth; project/experiment set via Hydra below) ----
unset WANDB_MODE   # in case a previous shell set WANDB_MODE=disabled

# ---- smoke-test vs full ----
SMOKE_TEST=${SMOKE_TEST:-1}
EXP_NAME=${EXP_NAME:-qwen2_5_7b_grpo_$(date +%Y%m%d_%H%M%S)}

# Knobs the upstream recipe reads from env vars (apply in both modes; smoke shrinks them below).
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}   # more thinking room for math; fits easily on H200
export ROLLOUT_TP=${ROLLOUT_TP:-2}                       # 4 DP replicas of vLLM x TP=2
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7} # H200 sweet spot per verl docs

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
  export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
  export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}  # 4 PPO steps per batch (512/128)
  export ROLLOUT_N=${ROLLOUT_N:-8}                       # bigger GRPO group -> lower advantage variance
  EXTRA_OVERRIDES=(
    trainer.save_freq=20
    trainer.test_freq=5
  )
fi

# Knobs the upstream recipe hardcodes -> override via Hydra. Tuned for 8x H200 141GB.
H200_TUNING=(
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384            # was 3000; H200 has plenty of HBM
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True     # cuts entropy peak memory
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True          # overlap comm with compute
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000     # forward-only, can be larger
  actor_rollout_ref.rollout.enable_chunked_prefill=True              # docs explicitly recommend
  actor_rollout_ref.rollout.max_num_batched_tokens=8192              # docs: ">2048" for big GPUs
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24000
  trainer.balance_batch=True                                         # DP load-balancing
)

# ---- log ----
mkdir -p run_logs
LOG=run_logs/${EXP_NAME}.log
echo "log: $LOG"
echo "wandb run: verl-test/${EXP_NAME}"
echo "smoke_test: ${SMOKE_TEST}"
echo "model: ${MODEL_PATH}"

# ---- launch ----
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  trainer.project_name=verl-test \
  trainer.experiment_name="$EXP_NAME" \
  trainer.default_local_dir=/dpc/kuin0100/hang/Documents/checkpoints/verl-test/$EXP_NAME \
  "${H200_TUNING[@]}" \
  "${EXTRA_OVERRIDES[@]}" \
  "$@" \
  2>&1 | tee "$LOG"
