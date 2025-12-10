import unittest
import sys
from unittest.mock import patch, MagicMock
from src.utils.compatibility import apply_backward_compatibility
import src.xgboost
import src.xgboost.model

class TestCompatibility(unittest.TestCase):
    def setUp(self):
        # Clean up modules if they exist to test clean application
        self._remove_modules()

    def tearDown(self):
        self._remove_modules()

    def _remove_modules(self):
        for mod in ['src.model', 'src.model.model', 'src.model.preprocessing']:
            if mod in sys.modules:
                del sys.modules[mod]

    def test_apply_backward_compatibility(self):
        # Action should not raise error
        try:
            apply_backward_compatibility()
        except Exception as e:
            self.fail(f"apply_backward_compatibility raised Exception: {e}")

    def test_apply_backward_compatibility_idempotent(self):
        # It should handle being called twice safely
        apply_backward_compatibility()
        apply_backward_compatibility()

