# func_weibull_fit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ==========================
# Core 2-parameter Weibull MLE
# ==========================
@dataclass
class WeibullFit:
    beta: float           # shape
    eta: float            # scale (same units as the input samples)
    n: int                # number of samples used in the fit
    loglik: float         # log-likelihood at optimum
    converged: bool
    iterations: int


def _initial_beta_guess(t: np.ndarray) -> float:
    # For Weibull, Var(ln T) = π^2 / (6 β^2)  => β ≈ π / (sqrt(6) * std(ln T))
    y = np.log(t)
    s = np.std(y, ddof=1)
    if not np.isfinite(s) or s <= 1e-9:
        return 2.0
    return float(np.pi / (np.sqrt(6.0) * s))


def fit_weibull_2p(
    samples: Iterable[float],
    *,
    max_iter: int = 100,
    tol: float = 1e-7,
    beta_bounds: Tuple[float, float] = (0.1, 20.0),
) -> Optional[WeibullFit]:
    """
    MLE for 2-parameter Weibull with **uncensored** samples.
    Returns None if fewer than 2 valid samples are available.
    """
    t = np.asarray(list(samples), dtype=float)
    t = t[np.isfinite(t) & (t > 0)]
    n = t.size
    if n < 2:
        return None

    # Precompute logs
    ln_t = np.log(t)
    s1 = ln_t.sum()

    # Initial beta
    beta = np.clip(_initial_beta_guess(t), beta_bounds[0], beta_bounds[1])

    def _g(beta_val: float) -> Tuple[float, float]:
        # Equation: g(β) = (Σ t^β ln t)/(Σ t^β) - (Σ ln t)/n - 1/β  = 0
        tb = t ** beta_val
        s3 = tb.sum()
        s2 = (tb * ln_t).sum()
        g = (s2 / s3) - (s1 / n) - (1.0 / beta_val)

        # Derivative g'(β)
        s2p = (tb * (ln_t ** 2)).sum()    # d/dβ Σ t^β ln t = Σ t^β (ln t)^2
        s3p = (tb * ln_t).sum()            # d/dβ Σ t^β = Σ t^β ln t
        dg = ((s2p * s3 - s2 * s3p) / (s3 ** 2)) + (1.0 / (beta_val ** 2))
        return g, dg

    converged = False
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        g, dg = _g(beta)
        step = g / dg
        beta_new = beta - step
        # keep in bounds & sane
        if not np.isfinite(beta_new):
            beta_new = max(beta_bounds[0], min(beta_bounds[1], beta * 0.9 + 1.0))
        beta_new = np.clip(beta_new, beta_bounds[0], beta_bounds[1])
        if abs(beta_new - beta) <= tol * max(1.0, beta):
            beta = float(beta_new)
            converged = True
            break
        beta = float(beta_new)

    # With β*, η̂ = ( (1/n) Σ t^β )^(1/β)
    tb = t ** beta
    eta = float((tb.mean()) ** (1.0 / beta))

    # Log-likelihood
    # ℓ = n ln β - n β ln η + (β-1) Σ ln t - Σ (t/η)^β
    loglik = n * np.log(beta) - n * beta * np.log(eta) + (beta - 1.0) * s1 - np.sum((t / eta) ** beta)

    return WeibullFit(beta=beta, eta=eta, n=n, loglik=float(loglik), converged=converged, iterations=iters)


# ==========================
# Build inter-arrival samples from events
# ==========================
def build_interarrival_days(
    df: pd.DataFrame,
    *,
    category_col: str,
    date_series: pd.Series,
    within_start: Optional[pd.Timestamp] = None,
    within_end: Optional[pd.Timestamp] = None,
    min_gap_days: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """
    For each breakdown category, compute inter-arrival times (in **days**) using event timestamps.
    Only consecutive **observed** gaps inside the [within_start, within_end] window are used.
    (No censoring terms are added; the first/last gaps spanning window edges are ignored.)

    Returns: {category -> np.ndarray of gap lengths in days (>0)}
    """
    # Normalize date series to pandas Timestamps
    ds = pd.to_datetime(date_series, errors="coerce")
    mask = ds.notna()
    if within_start is not None:
        mask &= (ds >= pd.Timestamp(within_start))
    if within_end is not None:
        mask &= (ds <= pd.Timestamp(within_end))

    df2 = df.loc[mask, [category_col]].copy()
    df2 = df2.assign(_ts=ds[mask]).dropna(subset=[category_col, "_ts"])
    if df2.empty:
        return {}

    out: Dict[str, np.ndarray] = {}
    for cat, grp in df2.groupby(category_col):
        g = grp.sort_values("_ts")["_ts"].drop_duplicates()
        if g.size < 2:
            continue
        gaps = g.diff().dt.total_seconds().iloc[1:] / 86400.0
        gaps = gaps[np.isfinite(gaps) & (gaps > min_gap_days)]
        if gaps.size > 0:
            out[str(cat)] = gaps.to_numpy(dtype=float)
    return out


# ==========================
# Fit per-category
# ==========================
def fit_weibull_by_category(
    df_classified: pd.DataFrame,
    *,
    category_col: str = "breakdown_category",
    date_series: pd.Series,
    within_start: Optional[pd.Timestamp] = None,
    within_end: Optional[pd.Timestamp] = None,
    min_samples: int = 2,
) -> pd.DataFrame:
    """
    Builds inter-arrival (days) per category and fits Weibull(η, β) via MLE.
    Returns a DataFrame with:
      ['Breakdown Category','n_intervals','weibull_beta','weibull_eta_days','converged','iterations','loglik']

    Notes:
    - Uses only **complete consecutive gaps** inside the chosen window (no censored contributions).
    - 'date_series' should align with df_classified's index (e.g., pulled from df_master[date_col] after filtering).
    """
    gaps_map = build_interarrival_days(
        df_classified,
        category_col=category_col,
        date_series=date_series,
        within_start=within_start,
        within_end=within_end,
    )

    rows: List[Dict] = []
    for cat, gaps in gaps_map.items():
        if len(gaps) < min_samples:
            rows.append({
                "Breakdown Category": cat,
                "n_intervals": int(len(gaps)),
                "weibull_beta": np.nan,
                "weibull_eta_days": np.nan,
                "converged": False,
                "iterations": 0,
                "loglik": np.nan,
                "note": "insufficient intervals",
            })
            continue
        fit = fit_weibull_2p(gaps)
        if fit is None:
            rows.append({
                "Breakdown Category": cat,
                "n_intervals": 0,
                "weibull_beta": np.nan,
                "weibull_eta_days": np.nan,
                "converged": False,
                "iterations": 0,
                "loglik": np.nan,
                "note": "fit failed",
            })
        else:
            rows.append({
                "Breakdown Category": cat,
                "n_intervals": int(fit.n),
                "weibull_beta": float(fit.beta),
                "weibull_eta_days": float(fit.eta),
                "converged": bool(fit.converged),
                "iterations": int(fit.iterations),
                "loglik": float(fit.loglik),
                "note": "",
            })
    if not rows:
        return pd.DataFrame(columns=[
            "Breakdown Category","n_intervals","weibull_beta","weibull_eta_days","converged","iterations","loglik","note"
        ])
    out = pd.DataFrame(rows)
    # deterministic order
    out = out.sort_values(["n_intervals","Breakdown Category"], ascending=[False, True]).reset_index(drop=True)
    return out


# ==========================
# Merge into category_summary
# ==========================
def apply_weibull_to_summary(
    category_summary: pd.DataFrame,
    weibull_fits: pd.DataFrame,
    *,
    beta_col_out: str = "Beta",
    eta_col_out: str = "Eta",
) -> pd.DataFrame:
    """
    Left-join Weibull results into the existing category_summary.
    Where available, overwrite Beta/Eta with fitted values:
      Beta  <- weibull_beta
      Eta   <- weibull_eta_days
    """
    if category_summary.empty or weibull_fits.empty:
        return category_summary
    
    # Check if required columns exist in weibull_fits
    required_cols = ["Breakdown Category", "weibull_beta", "weibull_eta_days"]
    missing_cols = [col for col in required_cols if col not in weibull_fits.columns]
    
    if missing_cols:
        # If columns are missing, return original category_summary without merging
        print(f"Warning: Missing columns in weibull_fits: {missing_cols}")
        print(f"Available columns: {list(weibull_fits.columns)}")
        return category_summary

    merged = category_summary.merge(
        weibull_fits[["Breakdown Category", "weibull_beta", "weibull_eta_days"]],
        on="Breakdown Category",
        how="left",
    )
    # Prefer fitted values when present
    beta = pd.to_numeric(merged["weibull_beta"], errors="coerce")
    eta = pd.to_numeric(merged["weibull_eta_days"], errors="coerce")

    merged[beta_col_out] = np.where(beta.notna(), beta, merged.get(beta_col_out))
    merged[eta_col_out] = np.where(eta.notna(), eta, merged.get(eta_col_out))

    # Drop helper cols
    merged = merged.drop(columns=[c for c in ["weibull_beta", "weibull_eta_days"] if c in merged.columns])
    return merged

