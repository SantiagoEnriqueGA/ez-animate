__version__ = "0.1.0"
from .animator import ClassificationAnimation, ForecastingAnimation, RegressionAnimation
from .utils import PCA, train_test_split

__all__ = [
    "ForecastingAnimation",
    "RegressionAnimation",
    "ClassificationAnimation",
    "PCA",
    "train_test_split",
]
