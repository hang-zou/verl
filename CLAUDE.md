# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Local fork of [verl](https://github.com/volcengine/verl), used for reinforcement-learning post-training (GRPO, DAPO) of LLMs and multimodal LLMs on NVIDIA H200 hardware. Primary models so far: Qwen 2.5 / Qwen 2.5-VL / Qwen 3 family. Upstream is tracked as the `upstream` remote; local changes live on the `dev` branch of `origin` (the user's fork at `hang-zou/verl`).

`AGENTS.md` (still in this directory) is the upstream verl-project contributor policy. That policy applies only if you're proposing changes back to `volcengine/verl` — it does **not** restrict local experimentation, running recipes, or modifying local copies of scripts in this fork.

## Repository notes

- `recipe/` is a git submodule (`verl-project/verl-recipe.git`). Run `git submodule update --init recipe` after a fresh checkout — this is where DAPO and other named algorithms live (`recipe/dapo/`, `recipe/r1/`, etc.).
- `launchers/` holds local wrapper scripts customized for this environment (see `launchers/README.md`).
- `verl/_telecomllm_patches.py` is auto-loaded by `import verl` (added to `verl/__init__.py`); it monkey-patches Megatron's `_set_attention_backend` to be non-asserting so the actor and ref can use different attention backends inside the same Ray worker.
- `run_logs/` and `install_logs/` are git-ignored — training and install output goes there.

## Environment

- **Conda env**: `verl` at `/dpc/kuin0100/hang/envs/verl` (Python 3.12).
  - `~/.condarc` already points `envs directories` at `/dpc/kuin0100/hang/envs/`, so `conda activate verl` works directly.
- **Hardware**: 1 node × 8× H200 (143 GB each). Driver advertises CUDA 13; verl wheels target CUDA 12.8.
- **Key package versions** (set by verl's install script — do not hand-tune): torch 2.8.0+cu128, vLLM 0.11.0, SGLang 0.5.2, flash-attn 2.8.1, flashinfer 0.3.1, Megatron-LM `core_v0.13.1`, TransformerEngine v2.6.

## Install / rebuild from scratch

The official `scripts/install_vllm_sglang_mcore.sh` is **not enough on its own** in this env — the system has no CUDA toolkit, cuDNN/NVTX live only as pip packages (not at `$CUDA_HOME/include`), the verl recipe path picks `mbridge` (ISEEKYAN, lowercase) not `megatron-bridge`, and TE 2.6 + Megatron Core 0.13.1 + Ray workers leave attention-backend env vars in mutually-incompatible states unless preset. The sequence below is what actually produces a runnable env. Install logs go to `install_logs/`. From this repo root.

### 1. Conda env

```bash
source /apps/ku/intel_h200_gpu/miniconda/3/etc/profile.d/conda.sh
conda create -y -n verl python=3.12 pip      # 3.12 matches TE-cu12 cp312 wheel + verl install.rst
conda activate verl
```

### 2. verl + its install script (Megatron + SGLang)

```bash
git submodule update --init recipe           # DAPO / r1 / etc. live here
USE_MEGATRON=1 USE_SGLANG=1 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .                   # editable verl install
```

This installs torch 2.8.0+cu128, vLLM 0.11.0, SGLang 0.5.2, flash-attn 2.8.1, flashinfer 0.3.1, Megatron-Core 0.13.1, and `transformer-engine` (Python piece). Inside this step **TE-torch source build will fail** with `nvidia.__file__ is None` / `cudnn.h: No such file` — expected. Don't hand-tune numpy/opencv mid-install; the trailing `opencv-fixer` step handles cross-package conflicts.

### 3. CUDA toolkit in the env (needed for TE-torch compile)

```bash
conda install -c nvidia -y cuda-toolkit=12.8   # gives us nvcc + headers in $CONDA_PREFIX
```

### 4. Symlink pip-installed cuDNN + NVTX so the compiler can find them

cuDNN/NVTX ship as pip packages under `site-packages/nvidia/{cudnn,nvtx}/`, not at `$CUDA_HOME/include`. Without these symlinks, TE-torch compile fails with `cudnn.h: No such file or directory` / `nvtx3/nvToolsExt.h: No such file or directory`.

```bash
ln -sfn /dpc/kuin0100/hang/envs/verl/lib/python3.12/site-packages/nvidia/nvtx/include/nvtx3        $CONDA_PREFIX/include/nvtx3
ln -sfn /dpc/kuin0100/hang/envs/verl/lib/python3.12/site-packages/nvidia/nvtx/include/nvToolsExt.h $CONDA_PREFIX/include/nvToolsExt.h
for f in /dpc/kuin0100/hang/envs/verl/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn*.h; do
  ln -sfn "$f" "$CONDA_PREFIX/include/$(basename $f)"
done
for f in /dpc/kuin0100/hang/envs/verl/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn*; do
  ln -sfn "$f" "$CONDA_PREFIX/lib/$(basename $f)"
done
```

### 5. `LD_LIBRARY_PATH` activate-hook (newer libstdc++ + cuDNN runtime libs)

Conda's `libstdc++.so.6.0.34` has `CXXABI_1.3.15` (needed by the compiled TE-torch `.so`); the system `/lib64/libstdc++.so.6` only goes up to `CXXABI_1.3.13`. Without this hook, `import transformer_engine.pytorch` fails with `CXXABI_1.3.15 not found`.

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/zz_ld_paths.sh <<'EOF'
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
EOF
conda deactivate && conda activate verl       # re-enter so the hook fires
```

### 6. TransformerEngine 2.6 — pre-built kernels + source-build torch frontend

```bash
export CUDA_HOME=$CONDA_PREFIX
export NVTE_FRAMEWORK=pytorch
export MAX_JOBS=32
pip install --no-deps --no-build-isolation \
    "transformer-engine-cu12==2.6.0.post1" \
    --extra-index-url https://pypi.nvidia.com/
pip install --no-deps --no-build-isolation -v \
    "transformer-engine-torch==2.6.0.post1" \
    --extra-index-url https://pypi.nvidia.com/  # source build, ~10–15 min
```

Don't use `pip install "transformer-engine[pytorch]==..."` with `--no-deps` — the `[pytorch]` extra is dropped along with the regular deps. Install the two sub-packages explicitly.

Verify:
```bash
python -c "import transformer_engine as te, transformer_engine.pytorch as tep, megatron.core; print('TE', te.__version__, 'OK')"
```

### 7. `mbridge` (ISEEKYAN) — for verl's `vanilla_mbridge=True` path

verl's `vanilla_mbridge=False` path uses NVIDIA's `megatron-bridge` package, which requires `megatron-core ≥ 0.14.0`; with our pinned 0.13.1 it fails on `ProcessGroupCollection`, `is_pp_first_stage`, `megatron.core.activations`, etc. We use the other branch instead.

```bash
pip install --no-deps mbridge                  # from PyPI; provides `mbridge.AutoBridge`
```

Then in **every Megatron recipe launch**, override:
```
actor_rollout_ref.actor.megatron.vanilla_mbridge=True
actor_rollout_ref.ref.megatron.vanilla_mbridge=True
```

### 8. Downgrade numpy to <2.3 (for numba at runtime)

The install script leaves `numpy` at `2.4.x` (opencv-python pulls it up, overriding the earlier `<2.0` pin). vLLM's worker processes import `numba`, which strictly requires `numpy < 2.3` and raises `ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.4.` at runtime. Every vLLM worker dies, engine core init fails, the whole job crashes ~2-3 min in.

```bash
pip install --no-deps "numpy>=2.0,<2.3"   # 2.2.6 satisfies numba + mistral-common + opencv-python
python -c "import numpy, numba; print(numpy.__version__, numba.__version__)"
```

`verl`/`megatron-core` pin `numpy<2.0` in metadata but use APIs that work fine on 2.x — pip will warn about the conflict; ignore.

### 9. NVTE attention-backend env vars (preset at every launch)

Megatron's `LanguageModule._set_attention_backend()` is **assertive**: once a model build sets `NVTE_FLASH_ATTN`/`NVTE_FUSED_ATTN`/`NVTE_UNFUSED_ATTN`, any subsequent model with a different backend crashes. mbridge defaults to `AttnBackend.auto` (which expects all three = `1`). Preset before launching any Megatron recipe:

```bash
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=1
export NVTE_UNFUSED_ATTN=1
```

(For recipes that explicitly use `attention_backend=flash`, set FUSED=0, UNFUSED=0 instead. Don't `unset` — TE/cuDNN/mbridge can set one to a stale value before the recipe's backend selector runs.)

### Sanity check the finished env

```bash
python -c "
import torch, vllm, transformers, ray, sglang, transformer_engine as te
import transformer_engine.pytorch as tep
import megatron.core, mbridge, verl
print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())
print('vllm', vllm.__version__, 'sglang', sglang.__version__, 'te', te.__version__)
print('mcore', getattr(megatron.core, '__version__', '?'), 'mbridge', getattr(mbridge, '__version__', '?'))
print('verl', verl.__version__ if hasattr(verl, '__version__') else 'editable')
"
```

## Backend choice (project-wide)

- **Training backend**: Megatron-LM (TP/PP/SP) or FSDP — FSDP wins on 7B single-node (1.4× faster), Megatron pays off at 32B+ or multi-node. Local launchers ship both flavors.
- **Inference (rollout) backend**: keep both vLLM and SGLang installed. Recipes default to `actor_rollout_ref.rollout.name=vllm`; override to `sglang` (or `trtllm`) to compare. Some recipes accept `INFER_BACKEND=vllm|sglang|trtllm` as an env var.

## Checkpoint output directory

Training checkpoints go to **`/dpc/kuin0100/hang/Documents/checkpoints/<project>/<experiment>/`**, NOT to `checkpoints/` inside this repo. All local launchers (`launchers/*.sh`) already wire this in via:

```
trainer.default_local_dir=/dpc/kuin0100/hang/Documents/checkpoints/verl-test/$EXP_NAME
```

When writing new launchers or running upstream recipes directly, include the same override so they don't fall back to the in-repo path. Each FSDP step-N checkpoint is ~86–93 GB for 7B (sharded model + fp32 optimizer + extras × 8 ranks); MLLM saves are slightly larger because of the vision encoder.

verl auto-saves the final state at the end of `total_training_steps` regardless of `save_freq` — to truly disable saves, set `trainer.save_freq` *and* watch out for the implicit end-of-run save.

## Models on disk

Pre-downloaded HuggingFace-format checkpoints — pass these as `MODEL_PATH=...` / `HF_MODEL_PATH=...` instead of using HF identifiers:

- `/dpc/kuin0100/hang/Documents/models/Qwen/Qwen2.5-7B-Instruct` — text 7B
- `/dpc/kuin0100/hang/Documents/models/Qwen/Qwen2.5-VL-7B-Instruct` — MLLM 7B (matches `examples/grpo_trainer/run_qwen2_5_vl_7b_megatron.sh`)
- `/dpc/kuin0100/hang/Documents/models/Qwen/Qwen2.5-VL-3B-Instruct` — MLLM 3B
- Other Qwen 2.5 (0.5B/1.5B/3B) and Qwen 3-4B (Instruct + Thinking) variants in the same dir, plus `meta-llama/Llama-3.1-8B`.
- Read-only borrowables at `/dpc/kuin0100/bohao/202509_InferenceModel/model/`: Qwen3.5-9B / 27B / 35B-A3B (MoE), DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Llama-70B.

## Where to find recipes

- **GRPO** — `examples/grpo_trainer/run_*.sh`
  - Text Megatron: `run_qwen3_8b_megatron.sh`, `run_qwen2-7b_math_megatron_fsdp.sh`
  - MLLM Megatron: `run_qwen2_5_vl_7b_megatron.sh`, `run_qwen3_vl_8b_megatron.sh`
- **DAPO** — `recipe/dapo/` (submodule). Canonical: `run_dapo_qwen2.5_32b.sh`. MLLM variants `run_dapo_qwen2.5_vl_{3b,7b,32b}_fsdp2_npu.sh` are tuned for Ascend NPUs — adapt rollout/training backend params for H200.
- **Other algorithms** also as recipes under `recipe/` (r1, prime, sppo, retool, swe_agent, ...) and `examples/{ppo_trainer,reinforce_plus_plus_trainer,rloo_trainer,...}/`.
- **Data preprocessing** — `examples/data_preprocess/{gsm8k,math_dataset,geo3k}.py` — emits parquet to `$HOME/data/<name>/`. Already pre-prepped for this env at `/dpc/kuin0100/hang/Documents/datasets/hf/`.

## Common commands

Activate and verify:
```bash
source /apps/ku/intel_h200_gpu/miniconda/3/etc/profile.d/conda.sh && conda activate verl
python -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.is_available())"
```

Local launchers live under `launchers/` (see `launchers/README.md` for the full table). Each accepts `SMOKE_TEST` (default 1) and any extra positional arg as a Hydra override.

Run text GRPO smoke test (Qwen2.5-7B, FSDP, H200-tuned):
```bash
bash launchers/qwen2_5_7b_fsdp.sh                          # SMOKE_TEST=1 by default
SMOKE_TEST=0 bash launchers/qwen2_5_7b_fsdp.sh             # full 15-epoch run
```

Run Megatron variant of the same:
```bash
bash launchers/qwen2_5_7b_megatron.sh                      # smoke
TP=2 GEN_TP=2 SMOKE_TEST=0 bash launchers/qwen2_5_7b_megatron.sh \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55  # tuned (TP=2 + lower vLLM mem util)
```

Run MLLM GRPO (Qwen2.5-VL-7B / Geo3K / FSDP2):
```bash
bash launchers/qwen2_5_vl_7b_fsdp.sh                       # smoke
SMOKE_TEST=0 bash launchers/qwen2_5_vl_7b_fsdp.sh          # full 15-epoch run
```

## Caveats

- The pinned `Megatron-LM` Python package (`core_v0.13.1`, installed by the install script) is **distinct** from any `Megatron-LM/` directory you may find in sibling project folders (e.g. `/dpc/kuin0100/hang/Documents/TelecomLLM/Megatron-LM/`). Don't conflate them.
- Most `recipe/dapo/run_dapo_*_npu.sh` scripts target Huawei Ascend NPUs; for NVIDIA GPUs use them as a config reference but swap backend-specific flags.
