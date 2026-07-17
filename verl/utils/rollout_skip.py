# RFLM fork compat: recipe modules (submodule, tracks a newer verl) import
# `from verl.utils.rollout_skip import RolloutSkip`; this fork hosts the
# implementation under verl.utils.skip.rollout_skip. The two versions differ
# in constructor signature, but every recipe call-site is gated on
# `actor_rollout_ref.rollout.skip_rollout` (absent -> False in this fork), so
# this alias only ever needs to satisfy the import. If you enable
# skip_rollout, reconcile the signatures first.
from verl.utils.skip.rollout_skip import AsyncRolloutSkip, RolloutSkip  # noqa: F401

__all__ = ["RolloutSkip", "AsyncRolloutSkip"]
