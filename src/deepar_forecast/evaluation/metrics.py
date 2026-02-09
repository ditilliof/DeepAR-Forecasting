"""
Evaluation metrics for probabilistic forecasting.

[REF_DEEPAR_PAPER] Section 4 - Evaluation methodology
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


def negative_log_likelihood(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: np.ndarray,
) -> float:
    """
    Compute negative log-likelihood under Student's t distribution.

    Primary metric for probabilistic models.
    [REF_STUDENT_T_LIKELIHOOD_TS]

    Args:
        y_true: Observed values (n_samples, horizon)
        mu: Predicted location (n_samples, horizon)
        sigma: Predicted scale (n_samples, horizon)
        nu: Predicted degrees of freedom (n_samples, horizon)

    Returns:
        Mean NLL across all samples and timesteps
    """
    log_likelihoods = []

    for i in range(y_true.shape[0]):
        for t in range(y_true.shape[1]):
            y = y_true[i, t]
            m = mu[i, t]
            s = sigma[i, t]
            n = nu[i, t]

            # Log PDF of Student's t
            log_prob = stats.t.logpdf(y, df=n, loc=m, scale=s)
            log_likelihoods.append(log_prob)

    nll = -np.mean(log_likelihoods)
    return nll


def continuous_ranked_probability_score(
    y_true: np.ndarray,
    samples: np.ndarray,
) -> float:
    """
    Compute CRPS (Continuous Ranked Probability Score).

    Proper scoring rule for probabilistic forecasts.
    Lower is better.

    CRPS = E[|X - Y|] - 0.5 * E[|X - X'|]
    where X, X' are independent samples, Y is the observation.

    Args:
        y_true: Observed values (n_obs, horizon)
        samples: Forecast samples (n_obs, n_samples, horizon)

    Returns:
        Mean CRPS across all observations and timesteps
    """
    n_obs = y_true.shape[0]
    horizon = y_true.shape[1]
    n_samples = samples.shape[1]

    crps_values = []

    for i in range(n_obs):
        for t in range(horizon):
            y = y_true[i, t]
            samp = samples[i, :, t]

            # E[|X - Y|]
            term1 = np.mean(np.abs(samp - y))

            # E[|X - X'|]  (pairwise differences)
            pairwise_diffs = []
            for j in range(n_samples):
                for k in range(j + 1, n_samples):
                    pairwise_diffs.append(np.abs(samp[j] - samp[k]))

            term2 = np.mean(pairwise_diffs) if pairwise_diffs else 0.0

            crps = term1 - 0.5 * term2
            crps_values.append(crps)

    return np.mean(crps_values)


def coverage_metrics(
    y_true: np.ndarray,
    quantiles: Dict[float, np.ndarray],
) -> Dict[str, float]:
    """
    Compute coverage metrics for predictive intervals.

    [REF_WALK_FORWARD_VALIDATION]

    Args:
        y_true: Observed values (n_obs, horizon)
        quantiles: Dict mapping quantile_level -> predictions (n_obs, horizon)
            Should include pairs like (0.1, 0.9), (0.025, 0.975) for intervals

    Returns:
        Dict with coverage percentages for different interval levels
    """
    results = {}

    # Common interval pairs
    intervals = [
        (0.25, 0.75, "50%"),  # 50% interval
        (0.1, 0.9, "80%"),  # 80% interval
        (0.025, 0.975, "95%"),  # 95% interval
    ]

    for lower_q, upper_q, label in intervals:
        if lower_q in quantiles and upper_q in quantiles:
            lower = quantiles[lower_q]
            upper = quantiles[upper_q]

            # Check if y_true falls within interval
            in_interval = (y_true >= lower) & (y_true <= upper)
            coverage = np.mean(in_interval) * 100  # Percentage

            results[f"coverage_{label}"] = coverage

    return results


def interval_width(quantiles: Dict[float, np.ndarray]) -> Dict[str, float]:
    """
    Compute average width of predictive intervals.

    Narrower intervals with good coverage are better (sharper predictions).

    Args:
        quantiles: Dict mapping quantile_level -> predictions

    Returns:
        Dict with interval widths
    """
    results = {}

    intervals = [
        (0.25, 0.75, "50%"),
        (0.1, 0.9, "80%"),
        (0.025, 0.975, "95%"),
    ]

    for lower_q, upper_q, label in intervals:
        if lower_q in quantiles and upper_q in quantiles:
            lower = quantiles[lower_q]
            upper = quantiles[upper_q]

            width = np.mean(upper - lower)
            results[f"width_{label}"] = width

    return results


def directional_accuracy(
    y_true: np.ndarray,
    median_pred: np.ndarray,
) -> float:
    """
    Compute directional accuracy (% of correct sign predictions).

    Args:
        y_true: Observed values
        median_pred: Median predictions

    Returns:
        Directional accuracy (0-1)
    """
    correct_signs = np.sign(y_true) == np.sign(median_pred)
    return np.mean(correct_signs)


def calibration_curve(
    y_true: np.ndarray,
    samples: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve (reliability diagram).

    For a well-calibrated model, predicted quantiles match empirical quantiles.

    Args:
        y_true: Observed values (n_obs, horizon) - flattened for analysis
        samples: Forecast samples (n_obs, n_samples, horizon)
        n_bins: Number of bins for calibration curve

    Returns:
        Tuple of (predicted_quantiles, empirical_quantiles)
    """
    # Flatten for analysis
    y_true_flat = y_true.flatten()
    n_obs = len(y_true_flat)

    # For each observation, compute empirical quantile within samples
    empirical_quantiles = []
    predicted_quantiles = np.linspace(0, 1, n_bins + 1)[:-1] + 0.5 / n_bins

    for i in range(n_obs):
        obs_idx = i // y_true.shape[1]
        time_idx = i % y_true.shape[1]

        y = y_true_flat[i]
        samp = samples[obs_idx, :, time_idx]

        # Empirical quantile: fraction of samples below observation
        empirical_q = np.mean(samp <= y)
        empirical_quantiles.append(empirical_q)

    # Bin empirical quantiles and compare to predicted
    binned_empirical = []
    for pred_q in predicted_quantiles:
        # Find observations with predicted quantile near pred_q
        bin_mask = (np.array(empirical_quantiles) >= pred_q - 0.5 / n_bins) & (
            np.array(empirical_quantiles) < pred_q + 0.5 / n_bins
        )

        if bin_mask.sum() > 0:
            binned_empirical.append(np.mean(np.array(empirical_quantiles)[bin_mask]))
        else:
            binned_empirical.append(pred_q)  # No data in bin, assume perfect calibration

    return predicted_quantiles, np.array(binned_empirical)


def point_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard point forecast metrics (RMSE, MAE, MAPE).

    Args:
        y_true: Observed values
        y_pred: Predicted values (typically median)

    Returns:
        Dict with RMSE, MAE, MAPE
    """
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())

    # MAPE (handle zero values)
    mape_values = []
    for yt, yp in zip(y_true.flatten(), y_pred.flatten()):
        if abs(yt) > 1e-8:
            mape_values.append(abs((yt - yp) / yt))

    mape = np.mean(mape_values) * 100 if mape_values else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    samples: np.ndarray,
    quantiles: Dict[float, np.ndarray],
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    nu: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: Observed values (n_obs, horizon)
        samples: Forecast samples (n_obs, n_samples, horizon)
        quantiles: Dict of quantile predictions
        mu, sigma, nu: Distribution parameters (for NLL)

    Returns:
        Dict with all metrics
    """
    logger.info("Computing evaluation metrics")

    metrics = {}

    # Point forecast metrics (using median)
    if 0.5 in quantiles:
        median_pred = quantiles[0.5]
        point_metrics = point_forecast_metrics(y_true, median_pred)
        metrics.update(point_metrics)

        # Directional accuracy
        metrics["directional_accuracy"] = directional_accuracy(y_true, median_pred)

    # CRPS
    crps = continuous_ranked_probability_score(y_true, samples)
    metrics["crps"] = crps

    # Coverage and interval width
    coverage = coverage_metrics(y_true, quantiles)
    metrics.update(coverage)

    widths = interval_width(quantiles)
    metrics.update(widths)

    # NLL (if distribution parameters provided)
    if mu is not None and sigma is not None and nu is not None:
        nll = negative_log_likelihood(y_true, mu, sigma, nu)
        metrics["nll"] = nll

    logger.info(f"Metrics: {metrics}")

    return metrics
