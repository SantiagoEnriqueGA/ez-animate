from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .classification_animation import ClassificationAnimation
from .clustering_animation import ClusteringAnimation
from .forecasting_animation import ForecastingAnimation
from .regression_animation import RegressionAnimation
from .transformation_animation import TransformationAnimation
from .utils import PCA, train_test_split

# Keep package version in sync with pyproject by reading distribution metadata.
# In editable/installed environments this will match [project].version.
try:  # Prefer distribution name as published on PyPI
    __version__ = _version("ez-animate")
except PackageNotFoundError:
    # Fallback for local source tree without installed metadata
    __version__ = "0.0.0"

__all__ = [
    "ForecastingAnimation",
    "RegressionAnimation",
    "ClassificationAnimation",
    "ClusteringAnimation",
    "TransformationAnimation",
    "PCA",
    "train_test_split",
]
