# API Reference

## Animation Classes
This section describes the main animation classes provided by `ez-animate`. Each class is designed to create specific types of animations for different use cases, such as regression, classification, clustering, forecasting, and data transformation.

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
    metric_fn=None,
    plot_metric_progression=False,
    max_metric_subplots=1,
    **kwargs
)
```

#### Parameters:
- `model`: Regression model class (e.g., `LinearRegression`, `SVR`).
- `X`: Feature matrix (2D array-like).
- `y`: Target vector (1D array-like).
- `test_size`: Fraction of data to use for testing (default: 0.3).
- `dynamic_parameter`: Parameter to vary dynamically (e.g., `C` for SVR).
- `static_parameters`: Dictionary of static parameters (e.g., `{'kernel': 'linear'}`).
- `keep_previous`: Whether to keep and display previous model states (default: False).
- `max_previous`: Maximum number of previous states to keep (default: None).
- `pca_components`: Number of PCA components to reduce dimensionality (default: 1).
- `metric_fn`: Optional metric function or list of functions (e.g., MSE, R2) to calculate and display during animation.
- `plot_metric_progression`: Whether to plot the progression of the metric over time.
- `max_metric_subplots`: Maximum number of subplots to show for metric progression (if multiple metrics).
- `**kwargs`: Additional keyword arguments for plot customization. See the "Customization with **kwargs**" section for details.


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
    metric_fn=None,
    plot_metric_progression=None,
    max_metric_subplots=1,
    **kwargs
)
```

#### Parameters:
- `model`: Classification model class (e.g., `LogisticRegression`, `SVC`).
- `X`: Feature matrix (2D array-like).
- `y`: Target vector (1D array-like).
- `test_size`: Fraction of data to use for testing (default: 0.3).
- `dynamic_parameter`: Parameter to vary dynamically (e.g., `C` for SVC).
- `static_parameters`: Dictionary of static parameters (e.g., `{'kernel': 'rbf'}`).
- `keep_previous`: Whether to keep and display previous model states (default: False).
- `scaler`: Scaler instance for preprocessing (e.g., `StandardScaler`).
- `pca_components`: Number of PCA components to reduce dimensionality (default: 2).
- `plot_step`: Step size for mesh grid in decision boundary plots (default: 0.02).
- `metric_fn`: Optional metric function or list of functions (e.g., accuracy, F1) to calculate and display during animation.
- `plot_metric_progression`: Whether to plot the progression of the metric over time.
- `max_metric_subplots`: Maximum number of subplots to show for metric progression (if multiple metrics).
- `**kwargs`: Additional keyword arguments for plot customization. See the "Customization with **kwargs**" section for details.



### ClusteringAnimation

Creates Matplotlib animations for clustering models (e.g., K-Means).

- Inherits from `AnimationBase`.
- Accepts a clustering model, input data, and optional true labels.
- Supports dynamic/static parameters, PCA, scaling, cluster center tracing, and plot customization.
- Returns a Matplotlib `FuncAnimation` object for display or saving.

**Constructor:**
```python
ClusteringAnimation(
    model,
    data,
    labels=None,
    test_size=0.3,
    dynamic_parameter=None,
    static_parameters=None,
    keep_previous=False,
    trace_centers=False,
    scaler=None,
    pca_components=2,
    metric_fn=None,
    plot_metric_progression=None,
    max_metric_subplots=1,
    **kwargs
)
```

#### Parameters:
- `model`: Clustering model class (e.g., `KMeans`, `DBSCAN`).
- `data`: Input data for clustering (2D array-like).
- `labels`: Optional true labels for coloring points (1D array-like or list).
- `test_size`: Fraction of data to use for testing (default: 0.3).
- `dynamic_parameter`: Parameter to vary dynamically (e.g., `n_clusters`).
- `static_parameters`: Dictionary of static parameters (e.g., `{'init': 'k-means++'}`).
- `keep_previous`: Whether to keep and display previous cluster assignments/centers (default: False).
- `trace_centers`: Whether to trace the movement of cluster centers over iterations (default: False).
- `scaler`: Scaler instance for preprocessing (e.g., `StandardScaler`).
- `pca_components`: Number of PCA components to reduce dimensionality (default: 2).
- `metric_fn`: Optional metric function or list of functions (e.g., silhouette_score) to calculate and display during animation.
- `plot_metric_progression`: Whether to plot the progression of metrics over frames.
- `max_metric_subplots`: Maximum number of metric subplots to display.
- `**kwargs`: Additional keyword arguments for plot customization. See the "Customization with **kwargs**" section for details.


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
    metric_fn=None,
    plot_metric_progression=None,
    max_metric_subplots=1,
    **kwargs
)
```

#### Parameters:
- `model`: Forecasting model class (e.g., `ARIMA`, `ExponentialSmoothing`).
- `train_series`: Training time series data (1D or 2D array-like).
- `test_series`: Test time series data (1D or 2D array-like).
- `forecast_steps`: Number of steps to forecast at each frame.
- `dynamic_parameter`: Parameter to vary dynamically (e.g., `order` for ARIMA).
- `static_parameters`: Dictionary of static parameters (e.g., `{'trend': 'add'}`).
- `keep_previous`: Whether to keep and display previous forecasts (default: False).
- `max_previous`: Maximum number of previous forecasts to keep (default: None).
- `metric_fn`: Optional metric function or list of functions (e.g., MSE, MAE) to calculate and display during animation.
- `plot_metric_progression`: Whether to plot the progression of metrics over frames.
- `max_metric_subplots`: Maximum number of metric subplots to display.
- `**kwargs`: Additional keyword arguments for plot customization. See the "Customization with **kwargs**" section for details.


### TransformationAnimation

Creates Matplotlib animations for data transformation steps (e.g., scaling, normalization, PCA).

- Inherits from `AnimationBase`.
- Accepts a transformer class and input data `X`.
- Supports dynamic/static parameters, optional metric tracking, and plot customization.
- Returns a Matplotlib `FuncAnimation` object for display or saving.

**Constructor:**
```python
TransformationAnimation(
  transformer,
  X,
  dynamic_parameter,
  static_parameters=None,
  keep_previous=False,
  metric_fn=None,
  plot_metric_progression=None,
  max_metric_subplots=1,
  **kwargs,
)
```

#### Parameters:
- `transformer`: Transformer class (e.g., `StandardScaler`, `MinMaxScaler`, `PCA`).
- `X`: Input data array (2D array-like).
- `dynamic_parameter`: Name of the transformer parameter to vary dynamically over frames (e.g., `n_components`, `feature_range`). Required.
- `static_parameters`: Dictionary of static parameters for the transformer (default: `None`).
- `keep_previous`: Whether to keep and display previous transformed states (default: `False`).
- `metric_fn`: Optional metric function or list of functions to calculate and display during animation. Each function should accept `(X_original, X_transformed)` and return a numeric value (e.g., reconstruction error, average distance).
- `plot_metric_progression`: Whether to plot the progression of the metric(s) over time.
- `max_metric_subplots`: Maximum number of subplots to show for metric progression (if multiple metrics).
- `**kwargs`: Additional keyword arguments for plot customization. See the "Customization with **kwargs**" section for details.


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

## Customization with `**kwargs`

You can customize the appearance of various plot elements by passing dictionaries of keyword arguments to the animation class constructor. These dictionaries are passed directly to the underlying Matplotlib plotting functions.

### Common Keyword Arguments
These `kwargs` can be used with any animation class.

#### General Plot Customization
- `title_kwargs`: Customizes the main plot title. Passed to `ax.set_title()`.
  - Example: `title_kwargs={'fontsize': 16, 'fontweight': 'bold'}`
- `suptitle_kwargs`: Customizes the figure's super-title. Passed to `fig.suptitle()`.
  - Example: `suptitle_kwargs={'fontsize': 20, 'y': 0.98}`
- `xlabel_kwargs`: Customizes the x-axis label. Passed to `ax.set_xlabel()`.
  - Example: `xlabel_kwargs={'fontsize': 12}`
- `ylabel_kwargs`: Customizes the y-axis label. Passed to `ax.set_ylabel()`.
  - Example: `ylabel_kwargs={'fontsize': 12}`
- `ax_kwargs`: Customizes general axes properties. Passed to `ax.tick_params()`.
  - Example: `ax_kwargs={'labelsize': 10}`
- `legend_kwargs`: Customizes the plot legend. Passed to `ax.legend()`.
  - Example: `legend_kwargs={'loc': 'lower right', 'fontsize': 'small'}`
- `grid_kwargs`: Customizes the plot grid lines. Passed to `ax.grid()`.
  - Example: `grid_kwargs={'linestyle': ':', 'alpha': 0.6}`

#### Metric Subplot Customization
These `kwargs` are used when `plot_metric_progression=True` in the animation class constructor. They customize the appearance of the metric subplots and their corresponding lines.

- `metric_title_kwargs`: Customizes the metric subplot title(s).
- `metric_line_kwargs`: Customizes the metric progression line(s).
- `metric_ax_kwargs`: Customizes the metric subplot axes.
- `metric_xlabel_kwargs`: Customizes the metric subplot x-axis label(s).
- `metric_ylabel_kwargs`: Customizes the metric subplot y-axis label(s).
- `metric_annotation_kwargs`: Customizes the text annotation for the metric value.

### Class-Specific Keyword Arguments
The following `kwargs` are used with specific animation classes. They customize the appearance of the corresponding plot elements.

#### RegressionAnimation
- `scatter_kwargs`: Customizes the scatter plot for training data points.
  - Example: `scatter_kwargs={'color': 'green', 's': 50}`
- `scatter_kwargs_test`: Customizes the scatter plot for test data points.
  - Example: `scatter_kwargs_test={'color': 'orange', 'marker': 's'}`
- `line_kwargs`: Customizes the fitted regression line.
  - Example: `line_kwargs={'color': 'purple', 'linewidth': 3}`

#### ClassificationAnimation
- `scatter_kwargs`: Customizes the scatter plot for training data points.
  - Example: `scatter_kwargs={'cmap': 'viridis', 's': 60}`
- `scatter_kwargs_test`: Customizes the scatter plot for test data points.
  - Example: `scatter_kwargs_test={'marker': 'D', 's': 60}`
- `decision_boundary_kwargs`: Customizes the filled decision boundary regions. Passed to `ax.contourf()`.
  - Example: `decision_boundary_kwargs={'alpha': 0.3, 'cmap': 'plasma'}`
- `decision_boundary_line_kwargs`: Customizes the decision boundary contour lines. Passed to `ax.contour()`.
  - Example: `decision_boundary_line_kwargs={'linewidths': 2, 'colors': 'k'}`

#### ClusteringAnimation
- `scatter_kwargs`: Customizes the scatter plot for clustered training data points.
- `scatter_kwargs_test`: Customizes the scatter plot for clustered test data points.
- `cluster_gray_train_kwargs`: Customizes the appearance of un-clustered training points.
- `cluster_gray_test_kwargs`: Customizes the appearance of un-clustered test points.
- `cluster_center_kwargs`: Customizes the cluster center markers.
  - Example: `cluster_center_kwargs={'marker': 'X', 's': 400, 'c': 'red'}`
- `prev_center_kwargs`: Customizes the markers for previous cluster centers (if `keep_previous=True`).
- `trace_line_kwargs`: Customizes the lines tracing cluster center movement (if `trace_centers=True`).
  - Example: `trace_line_kwargs={'linestyle': '-', 'linewidth': 1.5}`

#### ForecastingAnimation
- `train_line_kwargs`: Customizes the line plot for the training time series.
  - Example: `train_line_kwargs={'color': 'gray', 'label': 'Historical Data'}`
- `vline_kwargs`: Customizes the vertical line indicating the start of the forecast.
  - Example: `vline_kwargs={'color': 'red', 'linestyle': ':'}`
- `fitted_line_kwargs`: Customizes the line for the model's fitted values on the training data.
  - Example: `fitted_line_kwargs={'color': 'orange', 'zorder': 4}`
- `forecast_line_kwargs`: Customizes the line for the forecasted values.
  - Example: `forecast_line_kwargs={'color': 'purple', 'marker': 'o'}`

#### TransformationAnimation
- `scatter_kwargs`: Customizes the scatter plot for the original input data points.
  - Example: `scatter_kwargs={'color': 'blue', 'alpha': 0.6}`
- `scatter_kwargs_test`: Customizes the scatter plot for the transformed data points.
  - Example: `scatter_kwargs_test={'color': 'orange', 'alpha': 0.7, 'marker': 'x'}`
