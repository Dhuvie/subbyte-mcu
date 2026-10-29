"""
Test Split Quarantine Guard and Accounting Verification.
Enforces that model selection is conducted exclusively on validation splits,
and final test evaluation occurs only after configuration freezing.
"""

from pathlib import Path
from typing import Dict, List, Any

class TestSplitLockedError(Exception):
    """Raised when test metrics are accessed before an experiment configuration is frozen."""
    __test__ = False

class AccountingGuard:
    def __init__(self):
        self._frozen_configs = set()

    def freeze_config(self, config_hash: str):
        self._frozen_configs.add(config_hash)

    def is_frozen(self, config_hash: str) -> bool:
        return config_hash in self._frozen_configs

    def assert_can_evaluate_test(self, config_hash: str):
        if not self.is_frozen(config_hash):
            raise TestSplitLockedError(
                f"Evaluation on quarantined test set blocked: configuration '{config_hash}' is not frozen!"
            )

GLOBAL_GUARD = AccountingGuard()
