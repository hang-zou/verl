# TelecomLLM-local verl launchers

Thin wrappers around the upstream `examples/grpo_trainer/*.sh` recipes, customized for the TelecomLLM environment (8× H200, local model + dataset paths, `verl-test` wandb project, checkpoints under `/dpc/kuin0100/hang/Documents/checkpoints/`).

Every launcher:
- Activates the `verl` conda env.
- `cd`s to the verl repo root before running, so it works regardless of where you invoke it from.
- Generates an `EXP_NAME` (overridable) and routes the log to `verl/run_logs/${EXP_NAME}.log`.
- Sets `trainer.project_name=verl-test` and `trainer.default_local_dir=/dpc/kuin0100/hang/Documents/checkpoints/verl-test/${EXP_NAME}`.
- Honors `SMOKE_TEST=1` (default → small batch, 5 steps, no save/eval) vs `SMOKE_TEST=0` (recipe defaults / H200-tuned full run).
- Passes through any extra positional args as Hydra overrides.

## Table

| Script | Model | Dataset | Training backend | Rollout backend | Notable overrides |
|---|---|---|---|---|---|
| `qwen3_4b_fsdp.sh` | Qwen3-4B-Instruct-2507 | GSM8K | FSDP | vLLM (TP=2) | Recipe defaults, no extra tuning |
| `qwen2_5_7b_fsdp.sh` | Qwen2.5-7B-Instruct | GSM8K | FSDP | vLLM (TP=2) | H200 token budgets (`ppo_max_token_len_per_gpu=16384`, `log_prob_max_token_len_per_gpu=24000`, `gpu_memory_utilization=0.7`), `enable_chunked_prefill=True`, `rollout.n=8`, `balance_batch=True` |
| `qwen2_5_7b_megatron.sh` | Qwen2.5-7B-Instruct | GSM8K | Megatron (TP=4 default; override `TP=2 GEN_TP=2` for the tuned version) | vLLM | `vanilla_mbridge=True` (ISEEKYAN/mbridge), `NVTE_*_ATTN=1` defensive presets, save disabled (Megatron mbridge save bug). With `TP=2` also pass `actor_rollout_ref.rollout.gpu_memory_utilization=0.55` to avoid OOM. |
| `qwen2_5_vl_7b_fsdp.sh` | Qwen2.5-VL-7B-Instruct | Geo3K (hiyouga/geometry3k) | FSDP2 | vLLM (TP=2) | `data.image_key=images`, `model.use_fused_kernels=False` (FSDP2 + verl's fused LM-head path crashes with mixed Tensor/DTensor matmul) |

## Quick start

```bash
# Smoke (5 training steps, ~10 min, validates the pipeline end-to-end)
bash launchers/qwen2_5_7b_fsdp.sh

# Full run (15 epochs, real GRPO training curve)
SMOKE_TEST=0 bash launchers/qwen2_5_7b_fsdp.sh

# Named wandb run + extra Hydra override
EXP_NAME=lr_sweep_5e7 bash launchers/qwen2_5_7b_fsdp.sh \
    actor_rollout_ref.actor.optim.lr=5e-7
```

Watch live progress at `https://wandb.ai/<your-entity>/verl-test/runs/<run-id>` (the URL prints to the log a few minutes after launch).

## Things to know

- **Checkpoint disk cost.** Each FSDP step save is ~86–93 GB for 7B (sharded model + fp32 optimizer + extras × 8 ranks). The recipe defaults save every 20 steps; with 15 epochs that adds up. Use `trainer.save_freq=...` to thin them, or `trainer.max_actor_ckpt_to_keep=N` to keep only the last N. verl also implicitly saves a final-step checkpoint when `total_training_steps` is reached, regardless of `save_freq`.
- **NVTE patch.** The Megatron launcher depends on `verl/_telecomllm_patches.py` (auto-loaded by `import verl`) to neutralize the cross-build attention-backend assertion in megatron-core 0.13.1.
- **MLLM caveat.** The VL launcher bakes in `use_fused_kernels=False`. Remove that if upstream fixes the fused-LM-head DTensor path.
- **Training/install logs** go to `verl/run_logs/` and `verl/install_logs/`. Both are gitignored.
