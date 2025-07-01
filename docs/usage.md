# Usage

## Basic Example

This section will show how to use `ez-animate` to create a simple Matplotlib animation with minimal code. Example code and explanations will be provided here.

```python
TBD
```

## Scikit-learn Example

This section demonstrates how to use `ez-animate` with a Scikit-learn lasso regression model to create an animation of the model's predictions for alpha values ranging from 0.01 to 1.0.

```python
import numpy as np

from ez_animate import RegressionAnimation
from sega_learn.utils import Metrics, make_regression
from sklearn.linear_model import Lasso

# Generate synthetic regression data
X, y = make_regression(n_samples=1000, n_features=1)

# Create the animation using RegressionAnimation
animator = RegressionAnimation(
    model=Lasso,
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
    title="Lasso Regression Animation",
    xlabel="Feature Coefficient",
    ylabel="Target Value",
    legend_loc="upper left",
    grid=True,
    figsize=(12, 6),
)

# Create and show the animation
alpha_range = np.arange(0.01, 1.0, 0.01)
animator.animate(frames=alpha_range, interval=150, blit=True, repeat=False)
animator.show()
```
![Lasso Regression Animation](plots/animator_lasso.gif)



## More Examples Coming Soon

- Animating a 2D line plot
- Customizing figures and axes
- Saving animations

See [API Reference](api.md) for details on available classes and functions.
