"""TelecomLLM-local Megatron patches.

Loaded by `import verl` so it auto-applies in every Ray worker (the main process and
all sub-actors all `import verl` early).

Currently patches:

1. `megatron.core.models.common.language_module.language_module.LanguageModule._set_attention_backend`

   Upstream asserts that the runtime env vars `NVTE_FLASH_ATTN` / `NVTE_FUSED_ATTN` /
   `NVTE_UNFUSED_ATTN` match the *model's* configured attention backend. This crashes
   whenever verl builds two Megatron models with different backends in the same Ray
   worker (actor via mbridge → `AttnBackend.auto`; ref via the non-vanilla path →
   `AttnBackend.flash`). The error message even tells you to "unset NVTE_FLASH_ATTN,
   NVTE_FUSED_ATTN and NVTE_UNFUSED_ATTN" — but verl itself sets one of them during the
   first model build, so the second build always asserts.

   The fix is to make the setter *write-only*: each model writes the env vars its
   backend expects; the next forward pass reads them at kernel-launch time. This is
   safe because verl never runs two Megatron model forwards concurrently in the same
   process — they're serialized through Ray's worker dispatch.
"""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)


def _patch_megatron_attention_backend() -> None:
    try:
        from megatron.core.models.common.language_module import language_module as _lm
        from megatron.core.transformer.enums import AttnBackend
    except Exception as e:  # noqa: BLE001
        _logger.debug("megatron-core not importable; skipping NVTE attn patch: %s", e)
        return

    # backend -> (NVTE_FLASH_ATTN, NVTE_FUSED_ATTN, NVTE_UNFUSED_ATTN)
    table = {
        AttnBackend.local: ("0", "0", "0"),
        AttnBackend.flash: ("1", "0", "0"),
        AttnBackend.fused: ("0", "1", "0"),
        AttnBackend.unfused: ("0", "0", "1"),
        AttnBackend.auto: ("1", "1", "1"),
    }

    def _patched_set_attention_backend(self) -> None:
        backend = self.config.attention_backend
        if backend not in table:
            return
        f, fu, un = table[backend]
        os.environ["NVTE_FLASH_ATTN"] = f
        os.environ["NVTE_FUSED_ATTN"] = fu
        os.environ["NVTE_UNFUSED_ATTN"] = un

    # Idempotent — safe to re-import.
    if getattr(_lm.LanguageModule._set_attention_backend, "_telecomllm_patched", False):
        return
    _patched_set_attention_backend._telecomllm_patched = True  # type: ignore[attr-defined]
    _lm.LanguageModule._set_attention_backend = _patched_set_attention_backend  # type: ignore[assignment]
    _logger.info("Applied TelecomLLM Megatron _set_attention_backend patch (write-only, non-asserting).")


_patch_megatron_attention_backend()
