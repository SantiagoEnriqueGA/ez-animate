import importlib
import os
import sys
import unittest
from importlib.metadata import PackageNotFoundError
from unittest import mock

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.utils import BaseTest

from ez_animate import *


class TestImports(BaseTest):
    """Tests that the main package can be imported correctly."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting Package Imports", end="", flush=True)

    def test_all_imports(self):
        """Test that all expected imports are available in the package."""
        import ez_animate

        self.assertIsNotNone(ez_animate)
        self.assertTrue(hasattr(ez_animate, "__version__"))

        # Animation Classes
        self.assertTrue(hasattr(ez_animate, "ClassificationAnimation"))
        self.assertTrue(hasattr(ez_animate, "ForecastingAnimation"))
        self.assertTrue(hasattr(ez_animate, "RegressionAnimation"))
        self.assertTrue(hasattr(ez_animate, "ClusteringAnimation"))
        self.assertTrue(hasattr(ez_animate, "TransformationAnimation"))

        # Utility Functions
        self.assertTrue(hasattr(ez_animate, "PCA"))
        self.assertTrue(hasattr(ez_animate, "train_test_split"))

    def test_all_imports_from_init(self):
        """Test that all expected attributes are imported in __init__.py."""
        import ez_animate

        expected_all = [
            "ForecastingAnimation",
            "RegressionAnimation",
            "ClassificationAnimation",
            "ClusteringAnimation",
            "TransformationAnimation",
            "PCA",
            "train_test_split",
        ]
        self.assertListEqual(expected_all, ez_animate.__all__)

    def test_imports_from_animator(self):
        """Test that animation classes can be imported correctly."""
        from ez_animate import (
            ClassificationAnimation,
            ClusteringAnimation,
            ForecastingAnimation,
            RegressionAnimation,
        )

        self.assertIsNotNone(ClassificationAnimation)
        self.assertIsNotNone(ForecastingAnimation)
        self.assertIsNotNone(RegressionAnimation)
        self.assertIsNotNone(ClusteringAnimation)
        self.assertIsNotNone(TransformationAnimation)

    def test_imports_from_utils(self):
        """Test that utility functions can be imported correctly."""
        from ez_animate.utils import PCA, train_test_split

        self.assertIsNotNone(PCA)
        self.assertIsNotNone(train_test_split)

    def test_imports_from_widcard(self):
        """Test that wildcard imports work correctly."""
        self.assertIsNotNone(ForecastingAnimation)
        self.assertIsNotNone(RegressionAnimation)
        self.assertIsNotNone(ClassificationAnimation)
        self.assertIsNotNone(ClusteringAnimation)
        self.assertIsNotNone(TransformationAnimation)
        self.assertIsNotNone(PCA)
        self.assertIsNotNone(train_test_split)

    def test_version_fallback_on_missing_metadata(self):
        """When importlib.metadata.version raises PackageNotFoundError, __version__ should be '0.0.0'."""
        original_module = sys.modules.get("ez_animate")
        try:
            # Remove cached module so import executes __init__ again under the patch
            if "ez_animate" in sys.modules:
                del sys.modules["ez_animate"]

            with mock.patch(
                "importlib.metadata.version", side_effect=PackageNotFoundError
            ):
                ez_animate = importlib.import_module("ez_animate")
                self.assertEqual(ez_animate.__version__, "0.0.0")
        finally:
            # Restore original module (if it existed) to avoid affecting other tests
            if original_module is not None:
                sys.modules["ez_animate"] = original_module
            else:
                # Ensure a clean state if we created a temporary module instance
                sys.modules.pop("ez_animate", None)


if __name__ == "__main__":
    unittest.main()
