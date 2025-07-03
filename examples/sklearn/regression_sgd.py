import numpy as np

from ez_animate import RegressionAnimation
from sega_learn.utils import Metrics, make_regression
from sklearn.linear_model import SGDRegressor

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
    model=SGDRegressor,
    X=X,
    y=y,
    test_size=0.25,
    dynamic_parameter="max_iter",
    static_parameters={"fit_intercept": True, "eta0": 0.0005},
    keep_previous=True,
    max_previous=25,
    metric_fn=[
        Metrics.mean_absolute_error,
        Metrics.r_squared,
        Metrics.mean_squared_error,
    ],
    plot_metric_progression=True,
)

# Set up the plot
animator.setup_plot(
    title="SGD Regression Animation",
    xlabel="Feature Coefficient",
    ylabel="Target Value",
    legend_loc="upper left",
    grid=True,
    figsize=(12, 6),
)

# Create and save the animation
alpha_range = np.arange(1, 100, 1)
animator.animate(frames=alpha_range, interval=150, blit=False, repeat=True)
# animator.save(
#     filename="animator_sgd.gif",
#     writer="pillow",
#     fps=10,
#     dpi=300,
# )

animator.show()
