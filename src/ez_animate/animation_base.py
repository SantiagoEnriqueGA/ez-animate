import sys
from abc import ABC, abstractmethod

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Animation Class
# The goal is to create a reusable and modular animation class that can handle animations for any model and dataset.

# Requirements
#   Modularity:          The class should be reusable for different models and datasets.
#                        Should have base class and subclasses for specific models types (regression, classification, forecasting).
#   Customizability:     Allow users to customize plot elements (e.g., colors, labels, titles).
#   Ease of Use:         Provide a simple interface for creating animations.
#   Support for Metrics: Include functionality to calculate and display metrics like MSE.
#   Saving Options:      Allow saving animations in different formats (e.g., GIF, MP4).
#   Dynamic Updates:     Support dynamic updates of model parameters (e.g., window size).
#   Plot Styling:        Provide options for grid, legends, axis limits, etc.


# High-level Design
#   Base Class:         AnimationBase
#     - Common attributes and methods for all animations.
#     - Methods for setting up the plot, updating the plot, and saving the animation.
#     - Abstract methods for model-specific updates (e.g., update_model, update_plot).
#   Subclasses:         RegressionAnimation, ClassificationAnimation, ForecastingAnimation
#     - Inherit from AnimationBase and implement model-specific updates.
#     - Each subclass can have its own attributes and methods specific to the model type.


class AnimationBase(ABC):
    """Base class for creating animations of machine learning models."""

    def __init__(
        self,
        model,
        train_series,
        test_series,
        dynamic_parameter=None,
        static_parameters=None,
        keep_previous=None,
        metric_fn=None,
        **kwargs,
    ):
        """Initialize the animation base class.

        Args:
            model: The forecasting model or any machine learning model.
            train_series: Training time series data.
            test_series: Testing time series data.
            dynamic_parameter: The parameter to update dynamically (e.g., 'window', 'alpha', 'beta').
            static_parameters: Static parameters for the model.
                Should be a dictionary with parameter names as keys and their values.
            keep_previous: Whether to keep all previous lines with reduced opacity.
            metric_fn: Optional metric function or list of functions (e.g., MSE) to calculate and display during animation.
            **kwargs: Additional customization options (e.g., colors, line styles).
        """
        # Input validation
        if train_series is None or test_series is None:
            raise ValueError("train_series and test_series must be provided.")
        if dynamic_parameter is None:
            raise ValueError("dynamic_parameter must be provided.")
        if not isinstance(static_parameters, (dict, type(None))):
            raise ValueError("static_parameters must be a dictionary or None.")
        if not isinstance(keep_previous, bool):
            raise ValueError("keep_previous must be a boolean.")

        self.model = model
        self.train_data = train_series
        self.test_data = test_series
        self.dynamic_parameter = dynamic_parameter  # Parameter to update dynamically
        self.static_parameters = (
            static_parameters if static_parameters is not None else {}
        )
        self.keep_previous = keep_previous
        self.kwargs = kwargs

        # Optional metric function (e.g., MSE)
        self.metric_fn = metric_fn
        # If self.metric_fn is not a list, convert it to a list
        if self.metric_fn and not isinstance(self.metric_fn, list):
            self.metric_fn = [self.metric_fn]

        # Plot elements
        self.fig, self.ax = None, None
        self.lines = {}
        self.title = None

    def setup_plot(
        self, title, xlabel, ylabel, legend_loc="upper left", grid=True, figsize=(12, 6)
    ):
        """Set up the plot for the animation.

        Args:
            title: Title of the plot.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            legend_loc: Location of the legend.
            grid: Whether to show grid lines.
            figsize: Size of the figure.
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if legend_loc is not None:
            # self.ax.legend(loc=legend_loc)
            # Will call legend() in update_plot() to update the legend
            self.add_legend = True
        else:
            self.add_legend = False
        self.ax.grid(grid)
        plt.tight_layout()

    @abstractmethod
    def update_model(self, frame):
        """Abstract method to update the model for a given frame. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def update_plot(self, frame):
        """Abstract method to update the plot for a given frame.Must be implemented by subclasses."""
        raise NotImplementedError

    def animate(self, frames, interval=150, blit=True, repeat=False):
        """Create the animation.

        Args:
            frames: Range of frames (e.g., window sizes).
            interval: Delay between frames in milliseconds.
            blit: Whether to use blitting for faster rendering.
            repeat: Whether to repeat the animation.
        """

        def _update(frame):
            self.update_model(frame)
            return self.update_plot(frame)

        self.ani = animation.FuncAnimation(
            self.fig,
            _update,
            frames=frames,
            interval=interval,
            blit=blit,
            repeat=repeat,
        )

        return self.ani

    def save(self, filename, writer="pillow", fps=5, dpi=100):
        """Save the animation to a file.

        Args:
            filename: Path to save the animation.
            writer: Writer to use (e.g., 'pillow' for GIF).
            fps: Frames per second.
            dpi: Dots per inch for the saved figure.
        """
        if not hasattr(self, "ani"):
            raise RuntimeError("Animation has not been created. Call `animate` first.")
        # print(f"Saving animation to {filename} (this may take a while)...")
        # progress_callback = lambda i, n: print(f"Saving frame {i+1}/{n}", end='\r')

        try:
            self.ani.save(filename, writer=writer, fps=fps, dpi=dpi)
            sys.stdout.write("\033[K")  # Clear the line
            print(f"Animation saved successfully to {filename}.")
        except Exception as e:
            sys.stdout.write("\033[K")  # Clear the line on error too
            print(f"\nError saving animation: {e}")

    def show(self):
        """Display the animation."""
        if not hasattr(self, "ani") or self.ani is None:
            raise RuntimeError("Animation has not been created. Call `animate` first.")
        if self.fig is None:
            raise RuntimeError("Plot has not been set up. Call `setup_plot` first.")

        try:
            plt.show()
            print("Animation displayed.")
        except Exception as e:
            print(f"Error showing animation: {e}")
            # Attempt to close the figure if it exists, in case plt.show failed partially
            if self.fig:
                plt.close(self.fig)
