"""Regression checks for the Phase 0 production-import boundary."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_legacy_imports import violations  # noqa: E402


class LegacyIsolationTest(unittest.TestCase):
    def test_direct_imports_are_rejected(self):
        self.assertTrue(violations("import legacy.physics"))
        self.assertTrue(violations("from sim_v3.engine import step"))

    def test_dynamic_imports_are_rejected(self):
        self.assertTrue(violations("__import__('legacy.engine')"))
        self.assertTrue(violations("importlib.import_module('sim_v3.engine')"))

    def test_unrelated_imports_are_allowed(self):
        self.assertEqual(violations("import math\nfrom bcod_sim import __version__"), [])


if __name__ == "__main__":
    unittest.main()
