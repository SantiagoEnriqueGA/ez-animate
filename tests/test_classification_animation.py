import os
import sys
import unittest

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.utils import BaseTest, suppress_print

from ez_animate import (
    ClassificationAnimation,
)
from sega_learn.linear_models import LogisticRegression
from sega_learn.utils import (
    Metrics,
    make_classification,
)


class TestClassificationAnimation(BaseTest):
    """Unit test for the ClassificationAnimation class."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        """Initializes the test suite."""
        print("\nTesting ClassificationAnimation Class", end="", flush=True)
        mpl.use("Agg")

    def setUp(self):  # NOQA D201
        """Prepares each test."""
        # Generate synthetic classification data
        self.X, self.y = make_classification(
            n_samples=100,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_classes=2,
            random_state=42,
        )

    def test_init(self):
        """Test ClassificationAnimation initialization."""
        animator = ClassificationAnimation(
            model=LogisticRegression,
            X=self.X,
            y=self.y,
            test_size=0.25,
            dynamic_parameter="max_iter",
            static_parameters={"learning_rate": 0.001},
            keep_previous=True,
        )
        self.assertIsInstance(animator.X_train, np.ndarray)
        self.assertIsInstance(animator.y_train, np.ndarray)
        self.assertIsInstance(animator.X_test, np.ndarray)
        self.assertIsInstance(animator.y_test, np.ndarray)
        self.assertIsInstance(animator.xx, np.ndarray)
        self.assertIsInstance(animator.yy, np.ndarray)
        self.assertIsInstance(animator.unique_classes, np.ndarray)
        self.assertIsInstance(animator.colors, np.ndarray)

    def test_init_with_pca(self):
        """Test initialization with PCA for multi-feature data."""
        # Create multi-feature data
        X_multi = np.random.rand(100, 5)
        y_multi = np.random.randint(0, 2, 100)

        with suppress_print():
            animator = ClassificationAnimation(
                model=LogisticRegression,
                X=X_multi,
                y=y_multi,
                test_size=0.25,
                dynamic_parameter="max_iter",
                pca_components=2,
            )
        self.assertTrue(animator.needs_pca)
        self.assertIsNotNone(animator.pca_instance)
        self.assertEqual(
            animator.X_train.shape[1], 2
        )  # Should be reduced to 2 components

    def test_init_no_dynamic_parameter(self):
        """Test initialization with no dynamic parameter."""
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=self.X,
                y=self.y,
                test_size=0.25,
                dynamic_parameter=None,
                keep_previous=True,
            )

    def test_init_no_static_parameters(self):
        """Test initialization with no static parameters."""
        animator = ClassificationAnimation(
            model=LogisticRegression,
            X=self.X,
            y=self.y,
            test_size=0.25,
            dynamic_parameter="max_iter",
            static_parameters=None,
            keep_previous=True,
        )
        self.assertIsNotNone(animator)
        self.assertDictEqual(animator.static_parameters, {})

    def test_init_no_X(self):
        """Test initialization with no X."""
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=None,
                y=self.y,
                test_size=0.25,
                dynamic_parameter="max_iter",
                keep_previous=True,
            )

    def test_init_no_y(self):
        """Test initialization with no y."""
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=self.X,
                y=None,
                test_size=0.25,
                dynamic_parameter="max_iter",
                keep_previous=True,
            )

    def test_setup_plot(self):
        """Test setup_plot with valid parameters."""
        animator = ClassificationAnimation(
            model=LogisticRegression,
            X=self.X,
            y=self.y,
            test_size=0.25,
            dynamic_parameter="max_iter",
        )
        animator.setup_plot("Test Classification", "Feature 1", "Feature 2")
        self.assertIsNotNone(animator.fig)
        self.assertIsNotNone(animator.ax)
        self.assertEqual(animator.ax.get_title(), "Test Classification")
        self.assertGreater(len(animator.scatter_train_dict), 0)
        self.assertGreater(len(animator.scatter_test_dict), 0)
        plt.close(animator.fig)

    def test_update_model(self):
        """Test update_model with valid frame parameter."""
        animator = ClassificationAnimation(
            model=LogisticRegression,
            X=self.X,
            y=self.y,
            test_size=0.25,
            dynamic_parameter="max_iter",
            static_parameters={"learning_rate": 0.001},
        )
        with suppress_print():
            # Test with a specific max_iter value
            animator.update_model(1000)
        self.assertIsInstance(animator.model_instance, LogisticRegression)
        self.assertEqual(animator.model_instance.max_iter, 1000)
        self.assertEqual(animator.model_instance.learning_rate, 0.001)

    def test_update_plot_with_metrics(self):
        """Test update_plot with metrics."""
        animator = ClassificationAnimation(
            model=LogisticRegression,
            X=self.X,
            y=self.y,
            test_size=0.25,
            dynamic_parameter="max_iter",
            static_parameters={"learning_rate": 0.001},
            metric_fn=[Metrics.accuracy, Metrics.precision],
        )
        with suppress_print():
            animator.setup_plot("Test Classification", "Feature 1", "Feature 2")
            animator.update_model(1000)

            # Check that update_plot returns a tuple of artists
            artists = animator.update_plot(1000)

        self.assertIsInstance(artists, tuple)
        self.assertEqual(
            len(artists), 2
        )  # For binary classification, returns decision boundary and boundary lines
        plt.close(animator.fig)


if __name__ == "__main__":
    unittest.main()
