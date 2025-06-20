import numpy as np

from ez_animate import RegressionAnimation
from sega_learn.utils import Metrics, make_regression
from sklearn.linear_model import Ridge

# Generate synthetic regression data
X, y = make_regression(
    n_samples=1000,
    n_features=1,
    noise=1.25,
    random_state=42,
    tail_strength=10,
    bias=0.5,
)


# Create the animation using RegressionAnimation
animator = RegressionAnimation(
    model=Ridge,
    X=X,
    y=y,
    test_size=0.25,
    dynamic_parameter="alpha",
    static_parameters={"max_iter": 1, "fit_intercept": True},
    keep_previous=True,
    metric_fn=[
        Metrics.mean_squared_error,
        Metrics.mean_absolute_error,
        Metrics.r_squared,
    ],
)

# Set up the plot
animator.setup_plot(
    title="Ridge Regression Animation",
    xlabel="Feature Coefficient",
    ylabel="Target Value",
    legend_loc="upper left",
    grid=True,
    figsize=(12, 6),
)

# Create and save the animation
alpha_range = np.arange(0.01, 1.0, 0.01)
animator.animate(frames=alpha_range, interval=150, blit=True, repeat=False)
# animator.save(
#     filename="examples/utils/plots/animator_ridge.gif",
#     writer="pillow",
#     fps=10,
#     dpi=300,
# )

animator.show()
