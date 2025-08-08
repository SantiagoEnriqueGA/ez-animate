import os
import sys
import unittest
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler

# Ensure project root is on path (follow existing tests pattern)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.utils import BaseTest, suppress_print  # noqa: E402

from ez_animate import TransformationAnimation  # noqa: E402


class TestTransformationAnimation(BaseTest):
    """Unit tests for the TransformationAnimation class."""

    @classmethod
    def setUpClass(cls):  # NOQA: D201
        """Initialize matplotlib backend for headless testing."""
        print("\nTesting TransformationAnimation Class", end="", flush=True)
        mpl.use("Agg")

    def setUp(self):  # NOQA: D201
        """Create a simple 2D dataset for plotting/transforming."""
        self.X = np.random.rand(100, 2)

    def tearDown(self):  # NOQA: D201
        """Close all matplotlib figures after each test."""
        plt.close("all")

    def test_init_valid(self):
        """Constructor sets fields and accepts valid args."""
        animator = TransformationAnimation(
            transformer=StandardScaler,
            X=self.X,
            dynamic_parameter="with_mean",
            static_parameters={"with_std": True},
            keep_previous=True,
        )
        self.assertTrue(hasattr(animator, "X"))
        self.assertIsNone(animator.transformed_X)
        self.assertTrue(hasattr(animator, "previous_transforms"))
        self.assertIsInstance(animator.previous_transforms, list)

    def test_init_invalid_args(self):
        """Constructor raises on invalid inputs."""
        with self.assertRaises(ValueError):
            TransformationAnimation(
                transformer=StandardScaler, X=None, dynamic_parameter="with_mean"
            )

        # transformer without fit
        class NoFit:
            pass

        with self.assertRaises(ValueError):
            TransformationAnimation(
                transformer=NoFit(), X=self.X, dynamic_parameter="p"
            )
        with self.assertRaises(ValueError):
            TransformationAnimation(
                transformer=StandardScaler, X=self.X, dynamic_parameter=123
            )
        with self.assertRaises(ValueError):
            TransformationAnimation(
                transformer=StandardScaler,
                X=self.X,
                dynamic_parameter="with_mean",
                static_parameters=["oops"],
            )
        with self.assertRaises(ValueError):
            TransformationAnimation(
                transformer=StandardScaler,
                X=self.X,
                dynamic_parameter="with_mean",
                keep_previous="yes",
            )

    def test_setup_plot(self):
        """setup_plot initializes fig/ax and draws original data scatter."""
        animator = TransformationAnimation(
            transformer=StandardScaler, X=self.X, dynamic_parameter="with_mean"
        )
        animator.add_legend = True
        animator.setup_plot("Transform", "F1", "F2", legend_loc="upper right")
        self.assertIsNotNone(animator.fig)
        self.assertIsNotNone(animator.ax)
        self.assertEqual(animator.ax.get_title(), "Transform")
        # One scatter for original data
        self.assertGreaterEqual(len(animator.ax.collections), 1)

    def test_setup_plot_no_legend(self):
        """No legend requested should not attach a legend."""
        animator = TransformationAnimation(
            transformer=StandardScaler, X=self.X, dynamic_parameter="with_mean"
        )
        animator.setup_plot("NoLegend", "F1", "F2", legend_loc=None)
        # After setup: no legend
        self.assertIsNone(animator.ax.get_legend())
        # After update_plot: still no legend
        with suppress_print():
            animator.update_model(True)
            animator.update_plot(True)
        self.assertIsNone(animator.ax.get_legend())

    def test_update_model_and_plot_no_metric(self):
        """update_model fits the transformer and update_plot returns artists tuple."""
        animator = TransformationAnimation(
            transformer=StandardScaler, X=self.X, dynamic_parameter="with_mean"
        )
        animator.setup_plot("Transform", "F1", "F2")
        with suppress_print():
            animator.update_model(True)
            artists = animator.update_plot(True)
        self.assertIsInstance(animator.transformer_instance, StandardScaler)
        self.assertEqual(animator.transformed_X.shape, self.X.shape)
        self.assertIsInstance(artists, tuple)
        self.assertEqual(len(artists), 1)  # only transformed scatter is returned
        # Title includes dynamic parameter and value
        self.assertIn("with_mean=True", animator.ax.get_title())

    def test_update_model_param_merge(self):
        """Dynamic and static params are merged into transformer instance."""
        animator = TransformationAnimation(
            transformer=StandardScaler,
            X=self.X,
            dynamic_parameter="with_mean",
            static_parameters={"with_std": True},
        )
        with suppress_print():
            animator.update_model(True)
        self.assertIs(animator.transformer_instance.with_mean, True)
        self.assertIs(animator.transformer_instance.with_std, True)

    def test_update_plot_twice_removes_previous(self):
        """Calling update_plot twice should replace the transformed scatter without piling up artists."""
        animator = TransformationAnimation(
            transformer=StandardScaler, X=self.X, dynamic_parameter="with_mean"
        )
        animator.setup_plot("Transform", "F1", "F2")
        with suppress_print():
            animator.update_model(True)
            _ = animator.update_plot(True)
            # Count: original + transformed
            count_after_first = len(animator.ax.collections)
            # Second update with a different frame value
            animator.update_model(False)
            _ = animator.update_plot(False)
            count_after_second = len(animator.ax.collections)
        # Should remain the same number of collections (original + latest transformed)
        self.assertEqual(count_after_first, count_after_second)

    def test_update_plot_with_metric(self):
        """update_plot computes metric(s) and updates title accordingly."""

        def mean_abs_shift(x, xt):
            return float(np.mean(np.abs(xt - x)))

        animator = TransformationAnimation(
            transformer=StandardScaler,
            X=self.X,
            dynamic_parameter="with_mean",
            metric_fn=[mean_abs_shift],
            plot_metric_progression=True,
        )
        animator.setup_plot("Metric", "F1", "F2")
        with suppress_print():
            animator.update_model(True)
            artists = animator.update_plot(True)
        self.assertIsInstance(artists, tuple)
        self.assertEqual(len(artists), 1)
        # Metric name appears in the title
        self.assertIn("mean_abs_shift:", animator.ax.get_title())

    def test_metric_progression_limits_to_max_subplots(self):
        """Only up to max_metric_subplots metric axes are created and tracked."""

        def m1(x, xt):
            return float(np.linalg.norm(xt - x))

        def m2(x, xt):
            return float(np.mean(xt))

        animator = TransformationAnimation(
            transformer=StandardScaler,
            X=self.X,
            dynamic_parameter="with_mean",
            metric_fn=[m1, m2],
            plot_metric_progression=True,
            max_metric_subplots=1,
        )
        animator.setup_plot("Metrics", "F1", "F2")
        # Only one metric axis/line should be created
        self.assertEqual(len(animator.metric_axes), 1)
        self.assertEqual(len(animator.metric_lines), 1)
        with suppress_print():
            animator.update_model(True)
            animator.update_plot(True)
            animator.update_model(False)
            animator.update_plot(False)
        # Only the first metric progression should be tracked and length should be 2
        self.assertEqual(len(animator.metric_progression), 1)
        self.assertEqual(len(animator.metric_progression[0]), 2)

    def test_single_metric_function_not_list(self):
        """A single metric function (not list) should be handled and shown in title without progression."""

        def mean_val(x, xt):
            return float(np.mean(xt))

        animator = TransformationAnimation(
            transformer=StandardScaler,
            X=self.X,
            dynamic_parameter="with_mean",
            metric_fn=mean_val,  # not a list
            plot_metric_progression=False,
        )
        animator.setup_plot("SingleMetric", "F1", "F2")
        with suppress_print():
            animator.update_model(True)
            _ = animator.update_plot(True)
        # Title should include metric
        self.assertIn("mean_val:", animator.ax.get_title())

    def test_set_kwargs_transformation_and_general(self):
        """_set_kwargs applies Transformation-specific and general kwargs."""
        animator = TransformationAnimation(
            transformer=StandardScaler, X=self.X, dynamic_parameter="with_mean"
        )
        custom_kwargs = {
            "ax_kwargs": {"fontsize": 14},
            "legend_kwargs": {"fontsize": 9},
            "title_kwargs": {"fontsize": 12},
            "xlabel_kwargs": {"fontsize": 8},
            "ylabel_kwargs": {"fontsize": 8},
            "grid_kwargs": {"alpha": 0.3},
            "scatter_kwargs": {"color": "purple", "alpha": 0.8, "zorder": 5},
            "scatter_kwargs_test": {"color": "cyan", "marker": "+", "zorder": 4},
        }
        animator._set_kwargs(**custom_kwargs, subclass="TransformationAnimation")
        # General
        self.assertEqual(animator.ax_kwargs["fontsize"], 14)
        self.assertEqual(animator.legend_kwargs["fontsize"], 9)
        self.assertEqual(animator.title_kwargs["fontsize"], 12)
        self.assertEqual(animator.xlabel_kwargs["fontsize"], 8)
        self.assertEqual(animator.ylabel_kwargs["fontsize"], 8)
        self.assertEqual(animator.grid_kwargs["alpha"], 0.3)
        # Transformation-specific
        self.assertEqual(animator.scatter_kwargs["color"], "purple")
        self.assertEqual(animator.scatter_kwargs["alpha"], 0.8)
        self.assertEqual(animator.scatter_kwargs["zorder"], 5)
        self.assertEqual(animator.scatter_kwargs_test["color"], "cyan")
        self.assertEqual(animator.scatter_kwargs_test["marker"], "+")
        self.assertEqual(animator.scatter_kwargs_test["zorder"], 4)

    @patch("matplotlib.animation.FuncAnimation")
    def test_animate(self, mock_animation):
        """Animate delegates to matplotlib FuncAnimation and stores ani."""
        animator = TransformationAnimation(
            transformer=StandardScaler, X=self.X, dynamic_parameter="with_mean"
        )
        animator.setup_plot("Animate", "F1", "F2")
        frames = [True, False, True]
        ani = animator.animate(frames=frames, interval=100, blit=True, repeat=False)
        self.assertEqual(ani, animator.ani)
        mock_animation.assert_called_once()


if __name__ == "__main__":
    unittest.main()
