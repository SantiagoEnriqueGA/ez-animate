import numpy as np

from ez_animate import TransformationAnimation
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic data (2D for visualization)
np.random.seed(42)
X = np.random.randn(1000, 2) * [2, 0.5] + [5, 10]

# Create TransformationAnimation instance
anim = TransformationAnimation(
    transformer=MinMaxScaler,
    X=X,
    dynamic_parameter="feature_range",
    static_parameters=None,
    keep_previous=False,
    scatter_kwargs={"color": "blue", "alpha": 0.6},
    scatter_kwargs_test={"color": "orange", "alpha": 0.7, "marker": "x"},
)

# Setup plot
anim.setup_plot(
    title="MinMaxScaler Transformation Animation",
    xlabel="Feature 1",
    ylabel="Feature 2",
    legend_loc="upper left",
    grid=True,
    figsize=(10, 6),
)

# Animate over different feature_range values
feature_ranges = [
    (0, 1),
    (-1, 1),
    (0, 0.5),
    (0.5, 1.5),
    (1, 2),
]
ani = anim.animate(frames=feature_ranges, interval=1000, blit=False, repeat=True)

# Show animation
anim.show()

# Optionally save animation as GIF
# anim.save("minmax_scaler_animation.gif", writer="pillow", fps=1, dpi=100)
