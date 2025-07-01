# API Reference

## Animation Classes
This section describes the main animation classes provided by `ez-animate`. Each class is designed to create specific types of animations for different use cases, such as regression, classification, and forecasting.

### RegressionAnimation

Creates Matplotlib animations for regression models.

- Inherits from `AnimationBase`.
- Accepts a regression model, feature matrix `X`, and target vector `y`.
- Supports dynamic and static model parameters, PCA, and plot customization.
- Returns a Matplotlib `FuncAnimation` object for display or saving.

**Constructor:**
```python
RegressionAnimation(
    model, 
    X, 
    y, 
    test_size=0.3, 
    dynamic_parameter=None,
    static_parameters=None,
    keep_previous=False,
    max_previous=None,
    pca_components=1,
    **kwargs
)
```

### ClassificationAnimation

Creates Matplotlib animations for classification models.
- Inherits from `AnimationBase`.
- Accepts a classification model, feature matrix `X`, and target vector `y`.
- Supports dynamic/static parameters, PCA, scaling, and plot customization.
- Returns a Matplotlib `FuncAnimation` object for display or saving.

**Constructor:**
```python
ClassificationAnimation(
    model, 
    X, 
    y, 
    test_size=0.3, 
    dynamic_parameter=None,
    static_parameters=None,
    keep_previous=False,
    scaler=None,
    pca_components=2,
    plot_step=0.02,
    **kwargs
)
```

### ForecastingAnimation

Creates Matplotlib animations for time series forecasting models.

- Inherits from `AnimationBase`.
- Accepts a forecasting model, training and test series, and forecast steps.
- Supports dynamic/static parameters and plot customization.
- Returns a Matplotlib `FuncAnimation` object for display or saving.

**Constructor:**
```python
ForecastingAnimation(
    model, 
    train_series, 
    test_series, 
    forecast_steps, 
    dynamic_parameter=None, 
    static_parameters=None, 
    keep_previous=False, 
    max_previous=None, 
    **kwargs
)
```


## Common Methods

All animation classes inherit the following methods:

### setup_plot
```python
setup_plot(title, xlabel, ylabel, legend_loc="upper left", grid=True, figsize=(12, 6))
```
Set up the Matplotlib figure and axes for the animation.

- `title`: Title of the plot.
- `xlabel`: X-axis label.
- `ylabel`: Y-axis label.
- `legend_loc`: Legend location (default: "upper left").
- `grid`: Show grid lines (default: True).
- `figsize`: Figure size (default: (12, 6)).

### animate
```python
animate(frames, interval=150, blit=True, repeat=False)
```
Create the animation using Matplotlib's `FuncAnimation`.

- `frames`: Range or iterable of frames.
- `interval`: Delay between frames in ms (default: 150).
- `blit`: Use blitting for faster rendering (default: True).
- `repeat`: Repeat the animation (default: False).

### save
```python
save(filename, writer="pillow", fps=5, dpi=100)
```
Save the animation to a file (e.g., GIF or MP4).

- `filename`: Output file path.
- `writer`: Animation writer (default: "pillow").
- `fps`: Frames per second (default: 5).
- `dpi`: Dots per inch (default: 100).

### show
```python
show()
```
Display the animation in a window or notebook.

