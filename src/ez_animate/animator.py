import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Animator:
    def __init__(self, fig, ax, plot_func, data_stream, **kwargs):
        """
        A simple animator class.
        
        Args:
            fig: Matplotlib figure object.
            ax: Matplotlib axes object.
            plot_func: A function that takes an axes and a data frame and plots it.
            data_stream: An iterable (e.g., list, generator) yielding data for each frame.
            **kwargs: Keyword arguments passed to FuncAnimation.
        """
        self.fig = fig
        self.ax = ax
        self.plot_func = plot_func
        self.data_stream = list(data_stream) # Realize the iterable
        self.kwargs = kwargs
        
    def _update(self, frame_num):
        self.ax.clear() # Simple clearing, can be optimized
        data_for_frame = self.data_stream[frame_num]
        self.plot_func(self.ax, data_for_frame)

    def create_animation(self):
        """Creates and returns the Matplotlib animation object."""
        return FuncAnimation(self.fig, self._update, frames=len(self.data_stream), **self.kwargs)