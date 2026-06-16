"""Probability calibration and reliability."""

from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from .features import MODEL_FEATURES
from .models import build_logistic


def calibrate_cv(
    train_df, method="isotonic", cv=5, balanced=True, features=MODEL_FEATURES
):
    """Cross-validated calibration over all training seasons."""
    model = CalibratedClassifierCV(build_logistic(balanced), method=method, cv=cv)
    model.fit(train_df[features], train_df["suppressed"])
    return model


def reliability(y_true, y_prob, n_bins=10):
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    return mean_pred, frac_pos
