# ez-animate

A high-level, declarative Python package for creating common Matplotlib animations with minimal boilerplate code.


## Table of Contents
- [Project Goals](#project-goals)
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
<!-- - [Full Documentation](#full-documentation) -->
- [Development/Contributing](#developmentcontributing)


## Project Goals

`ez-animate` aims to make it easy for data scientists, analysts, educators, and researchers to create standard Matplotlib animations quickly and with minimal code. It abstracts away the complexity of `FuncAnimation`, state management, and repetitive setup, letting you focus on your data and story.

### Why?
- **Complex Setup:** No need to write custom `init` and `update` functions.
- **State Management:** Simplifies handling data and artist states between frames.
- **Repetitive Code:** Reduces boilerplate for standard animations.

### Who is it for?
- **Primary:** Data scientists & analysts (exploratory analysis, presentations, notebooks)
- **Secondary:** Students, educators, and researchers (learning, teaching, publications)

## Features
- **Simple API:** Create animations with a few lines of code
- **Tested & Linted:** High code quality with `pytest` and `ruff`
- **Documentation:** Built with MkDocs (**TBD**)

## Installation

**Pending PyPI release. For now, install directly from GitHub**
```bash
pip install ez-animate
```


## Quickstart

```python
from ez_animate import RegressionAnimation

# Create and run the animation
animator = RegressionAnimation(
    model=Lasso,    # Scikit-learn or sega_learn model class
    X=X,
    y=y,
    test_size=0.25,
    dynamic_parameter="alpha",
    static_parameters={"max_iter": 1, "fit_intercept": True},
    keep_previous=True,
    metric_fn=Metrics.mean_squared_error,
)

# Set up the plot
animator.setup_plot(
    title="Regression Animation",
    xlabel="Feature Coefficient",
    ylabel="Target Value",
)

# Create the animation
animator.animate(frames=np.arange(0.01, 1.0, 0.01))

# Show and save the animation
animator.show()
animator.save("regression_animation.gif")
```

<!-- Update link here once live -->
<!-- ## Full Documentation

See the [MkDocs documentation](https://ez-animate.readthedocs.io/) for complete usage instructions, API references, and examples. -->

## Development/Contributing

See [DEVELOPMENT.md](DEVELOPMENT.md) for full development and contributing guidelines.


## Project Structure

```
ez-animate/
├─ .github/
│  ├─ ISSUE_TEMPLATE
│  └─ workflows
├─ examples/
│  ├─ plots
│  ├─ sega_learn
│  └─ sklearn
├─ src/
│  └─ ez_animate
└─ tests

```

## License

This project is licensed under the terms of the [MIT License](LICENSE).

## Dependency Management

- All dependencies are in `pyproject.toml`.
- Core: `[project.dependencies]`
- Other: `[project.optional-dependencies]`
- Use `uv pip install -e .[dev]` to install all dev tools.

## Acknowledgments

- Built with inspiration from the Matplotlib community.
- Thanks to all contributors!
