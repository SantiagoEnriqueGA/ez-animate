## API compatibility matrix

This guide summarizes the minimal method and shape expectations for models/transformers used by each animation class. Optional methods yield better visuals or UX; when missing, reasonable fallbacks are used.

### At‑a‑glance

| Animation class | Required methods | Shapes/assumptions | Best visuals and optional extras |
|---|---|---|---|
| RegressionAnimation | fit(X, y), predict(X) | X: (n_samples, n_features). If n_features > 1, an internal PCA reduces to 1D for plotting. | None required; optionally `keep_previous`/`max_previous` for trail effect. |
| ClassificationAnimation | fit(X, y), predict(X) | X: (n_samples, 2). If >2 features, internal PCA to 2D. Optional scaler may be applied by user. | For a crisp boundary line: `predict_proba(X_grid)` or `decision_function(X_grid)`. With labels only, filled regions are drawn; the boundary line may be omitted. |
| ClusteringAnimation | fit(X) and predict(X) or labels_ after fit | X: (n_samples, 2). If >2 features, internal PCA to 2D. Optional scaler may be applied by user. | `cluster_centers_` enables center markers. `trace_centers` works best when the number of centers is consistent across frames. |
| ForecastingAnimation | One of: fit(y_train) + forecast(steps) OR fit(X, y) + predict(k) or predict(X_future) | train_series/test_series: 1D arrays/Series. Outputs coerced to 1D numpy arrays. | Graceful fallbacks: if `predict(steps)` fails, `predict(X_future)` is attempted with a generated index. |
| TransformationAnimation | fit(X), transform(X) | X: (n_samples, n_features). Current visuals assume ≥2 features for 2D scatter. | dynamic_parameter (e.g., `n_components`) can drive the animation. |

---

## Details by Class

### RegressionAnimation

- **Required API**: `fit(X, y)`, `predict(X)`
- **Shapes**: `X` is `(n_samples, n_features)`. If `n_features` > 1, a lightweight PCA reduces to 1 component for plotting against the target.
- **Fallbacks**: n/a
- **Extras**: `keep_previous` with optional `max_previous` to create fading line trails. Multiple metrics supported; enable `plot_metric_progression` for side-panel plots.

### ClassificationAnimation

- **Required API**: `fit(X, y)`, `predict(X)`
- **Shapes**: assumes 2D inputs. If X has >2 features, an internal PCA reduces to 2D. You may pre-scale with your own scaler.
- Best visuals:
    - `decision_function(X_grid)` → contour at 0 level (preferred when available)
    - `predict_proba(X_grid)` → take positive-class probability and contour at 0.5
- **Fallbacks**: with only `predict`, filled decision regions are drawn from class labels; a separating contour line may be omitted.
- **Extras**: `keep_previous` fades older boundaries. Multiple metrics supported; enable `plot_metric_progression` to show trends.

### ClusteringAnimation

- **Required API**: `fit(X)` and either `predict(X)` or `labels_` set after fit.
- **Shapes**: 2D inputs; if more, internal PCA reduces to 2D.
- **Best visuals**: `cluster_centers_` to plot centers; `trace_centers` draws motion paths when the number of centers is consistent across frames.
- **Metrics**:
    - Unsupervised (X, labels): `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_score`
    - Supervised (y_true, labels): any callable metric if ground truth labels are available

### ForecastingAnimation

- **Required API** (one of):
    - `fit(y_train)` + `forecast(steps)`
    - `fit(X, y)` + `predict(k)` or `predict(X_future)`
- **Shapes**: train_series/test_series are 1D. Outputs are coerced to 1D numpy arrays for plotting.
- **Fallbacks**: if `predict(steps)` isn’t supported, a future index `X_future = np.arange(...).reshape(-1, 1)` is used with `predict(X_future)`.
- **Extras**: `keep_previous` for fading fitted/forecast lines; multiple metrics with optional progression panel.

### TransformationAnimation

- **Required API**: `fit(X)`, `transform(X)`
- **Shapes**: X should have ≥2 features for 2D scatter visuals. dynamic_parameter (e.g., n_components) drives the animation.
- **Metrics**: `callable metric(X_original, X_transformed)`. Progression panel supported.

---

## Tips and practical notes

- Metrics panel
    - Pass a list of metric callables; set `plot_metric_progression=True` to enable a right‑side panel with compact trend lines. Values are annotated per frame.
    - Control panel count with `max_metric_subplots`.

- Dimensionality reduction
    - Regression: automatically reduced to 1D for plotting when `n_features > 1`.
    - Classification/Clustering: automatically reduced to 2D when `n_features > 2`.

- Keep‑previous overlays
    - Set `keep_previous=True` for historical context with fading alpha; use `max_previous` to cap rendered artists for performance.

- Data types
    - numpy arrays are first‑class. train_test_split also supports pandas DataFrame/Series and scipy sparse matrices.

- Performance
    - Dense decision meshes can be slow; increase `plot_step` (e.g., 0.05–0.1) or restrict axis ranges to speed up rendering.

If your model/transformer doesn’t match these interfaces, a thin adapter that exposes the expected methods is usually enough to integrate with ez‑animate.
