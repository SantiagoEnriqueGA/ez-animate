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

    def test_init_with_invalid_Xy(self):
        """Test initialization with invalid X or y."""
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=None,
                y=self.y,
                test_size=0.25,
                dynamic_parameter="max_iter",
            )
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=self.X,
                y=None,
                test_size=0.25,
                dynamic_parameter="max_iter",
            )

    def test_init_with_invalid_test_size(self):
        """Test initialization with invalid test_size."""
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=self.X,
                y=self.y,
                test_size=-0.1,
                dynamic_parameter="max_iter",
            )
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=self.X,
                y=self.y,
                test_size=1.5,
                dynamic_parameter="max_iter",
            )

    def test_init_with_invalid_dynamic_parameter(self):
        """Test initialization with invalid dynamic_parameter."""
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=self.X,
                y=self.y,
                test_size=0.25,
                dynamic_parameter=0,
            )

    def test_init_with_invalid_static_parameters(self):
        """Test initialization with invalid static_parameters."""
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=self.X,
                y=self.y,
                test_size=0.25,
                dynamic_parameter="max_iter",
                static_parameters=["invalid_param"],
            )

    def test_init_with_invalid_keep_previous(self):
        """Test initialization with invalid keep_previous."""
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=self.X,
                y=self.y,
                test_size=0.25,
                dynamic_parameter="max_iter",
                keep_previous="invalid_value",
            )

    def test_init_with_invalid_pca_components(self):
        """Test initialization with invalid pca_components."""
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=self.X,
                y=self.y,
                test_size=0.25,
                dynamic_parameter="max_iter",
                pca_components=-1,
            )
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=self.X,
                y=self.y,
                test_size=0.25,
                dynamic_parameter="max_iter",
                pca_components=0,
            )

    def test_init_with_invalid_plot_step(self):
        """Test initialization with invalid plot_step."""
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=self.X,
                y=self.y,
                test_size=0.25,
                dynamic_parameter="max_iter",
                plot_step=-1,
            )
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=self.X,
                y=self.y,
                test_size=0.25,
                dynamic_parameter="max_iter",
                plot_step=0,
            )

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

    def test_init_with_pca_components(self):
        """Test initialization with PCA for multi-feature data for pca_components."""
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
                pca_components=3,
            )
        self.assertTrue(animator.needs_pca)
        self.assertIsNotNone(animator.pca_instance)
        self.assertEqual(
            animator.X_train.shape[1], 2
        )  # Should be reduced to 2 components

    def test_init_only_1_component(self):
        """Test initialization with PCA for multi-feature data with only 1 component raising an error."""
        # Create X.shape[1] < 2
        X_multi = np.random.rand(100, 1)
        y_multi = np.random.randint(0, 2, 100)
        with self.assertRaises(ValueError):
            ClassificationAnimation(
                model=LogisticRegression,
                X=X_multi,
                y=y_multi,
                test_size=0.25,
                dynamic_parameter="max_iter",
                pca_components=1,
            )

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

    def test_update_plot_with_metric(self):
        """Test update_plot with one metric."""
        animator = ClassificationAnimation(
            model=LogisticRegression,
            X=self.X,
            y=self.y,
            test_size=0.25,
            dynamic_parameter="max_iter",
            static_parameters={"learning_rate": 0.001},
            metric_fn=Metrics.accuracy,
        )
        with suppress_print():
            animator.setup_plot("Test Classification", "Feature 1", "Feature 2")
            animator.update_model(1000)

            # Check that update_plot returns a tuple of artists
            artists = animator.update_plot(1000)

        self.assertIsInstance(artists, tuple)
        self.assertEqual(len(artists), 2)

    def test_update_plot_model_no_predict_proba(self):
        """Test update_plot when model does not have predict_proba."""

        # Create a model that does not implement predict_proba
        class DummyModel:
            def __init__(self, **kwargs):
                pass

            def fit(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(X.shape[0])

        animator = ClassificationAnimation(
            model=DummyModel,
            X=self.X,
            y=self.y,
            test_size=0.25,
            dynamic_parameter="max_iter",
        )
        with suppress_print():
            animator.setup_plot("Test Classification", "Feature 1", "Feature 2")
            animator.update_model(1000)
            artists = animator.update_plot(1000)

        self.assertIsInstance(artists, tuple)
        self.assertEqual(len(artists), 2)

    def test_update_plot_model_no_predict(self):
        """Test update_plot when model does not have predict."""

        # Create a model that does not implement predict
        class DummyModel:
            def __init__(self, **kwargs):
                pass

            def fit(self, X, y):
                pass

            def predict_proba(self, X):
                return np.zeros((X.shape[0], 2))

        animator = ClassificationAnimation(
            model=DummyModel,
            X=self.X,
            y=self.y,
            test_size=0.25,
            dynamic_parameter="max_iter",
        )
        with suppress_print():
            animator.setup_plot("Test Classification", "Feature 1", "Feature 2")
            animator.update_model(1000)
            animator.model_instance.classes_ = np.arange(len(animator.unique_classes))
            artists = animator.update_plot(1000)

        self.assertIsInstance(artists, tuple)
        self.assertEqual(len(artists), 2)

        with suppress_print():
            animator.setup_plot("Test Classification", "Feature 1", "Feature 2")
            animator.update_model(1000)
            animator.model_instance.classes_ = np.arange(
                len(animator.unique_classes) + 1
            )
            artists = animator.update_plot(1000)

        self.assertIsInstance(artists, tuple)
        self.assertEqual(len(artists), 2)

    def test_update_plot_model_no_predict_or_predict_proba(self):
        """Test update_plot when model does not have predict or predict_proba."""

        # Create a model that does not implement predict or predict_proba
        class DummyModel:
            def __init__(self, **kwargs):
                pass

            def fit(self, X, y):
                pass

        animator = ClassificationAnimation(
            model=DummyModel,
            X=self.X,
            y=self.y,
            test_size=0.25,
            dynamic_parameter="max_iter",
        )
        with suppress_print() and self.assertRaises(AttributeError):
            animator.setup_plot("Test Classification", "Feature 1", "Feature 2")
            animator.update_model(1000)
            animator.update_plot(1000)


if __name__ == "__main__":
    unittest.main()
