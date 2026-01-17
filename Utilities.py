from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, UnivariateSpline
import statsmodels.api as sm


def build_mesh(c_pts, meta, n_moneyness=100, n_maturity=100):
    m_min = meta["x_min"].max()
    m_max = meta["x_max"].min()
    m_grid = np.linspace(m_min, m_max, n_moneyness)
    T_grid = np.linspace(c_pts["T"].min(), c_pts["T"].max(), n_maturity)
    return m_grid, T_grid


def interpolated_spline(
    df: pd.DataFrame,
    T_bin_width: float = 7 / 365,
    min_points: int = 6,
    use_pchip: bool = True,
    smooth_window: int = 5,
    spline_s: float = 0.1,
) -> Tuple[Dict[float, object], pd.DataFrame]:
    """
    Fit 1D interpolators of IV vs moneyness for each maturity bucket.

    Steps per maturity bucket:
      1) sort by moneyness
      2) average duplicates in moneyness
      3) optional rolling-median smoothing on IV
      4) fit either PCHIP or UnivariateSpline

    Returns:
      splines: dict keyed by T_mid (float) -> interpolator callable
      meta_df: DataFrame describing each fitted slice support
    """
    required_cols = {"moneyness", "T", "IV"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing columns: {sorted(missing)}")

    if T_bin_width <= 0:
        raise ValueError("T_bin_width must be > 0")
    if min_points < 2:
        raise ValueError("min_points must be >= 2")
    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1")

    work = df.loc[:, ["moneyness", "T", "IV"]].copy()
    work = work.dropna(subset=["moneyness", "T", "IV"])

    # Round Moneyness to nearest 1% to merge liquid/illiquid strikes
    work["moneyness"] = (work["moneyness"] * 100).round() / 100
    # Now group by this rounded moneyness and take the MAX vol (assuming peaks are real liquidity)
    work = work.groupby(["T", "moneyness"], as_index=False)["IV"].max()

    # Bucket maturities (weekly by default)
    work["T_bin"] = np.rint(work["T"].to_numpy() / T_bin_width).astype(int)

    splines: Dict[float, object] = {}
    meta_rows: list[dict] = []

    for T_bin, g in work.groupby("T_bin", sort=False):
        if len(g) < min_points:
            continue

        T_mid = float(np.nanmedian(g["T"].to_numpy()))

        x = g["moneyness"].to_numpy(dtype=float)
        y = g["IV"].to_numpy(dtype=float)

        # sort by x
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # collapse duplicates in x by averaging y
        # (vectorized grouping without an extra full DataFrame when possible)
        uniq_x, inv = np.unique(x, return_inverse=True)
        if uniq_x.size != x.size:
            y_sum = np.bincount(inv, weights=y)
            y_cnt = np.bincount(inv)
            x = uniq_x
            y = y_sum / y_cnt

        if x.size < min_points:
            continue

        if len(y) > 0:
            # 1. Separate the curve into left (ITM) and right (OTM) of ATM
            atm_mask = x >= 1.0

            # 2. Only enforce on the right side (OTM Calls)
            y_right = y[atm_mask]

            # 3. Apply "minimum so far" filter (accumulated minimum)
            # If y drops to 0.33, it can never go back up to 0.34 later.
            # if len(y_right) > 0:
            #     y_monotonic = np.minimum.accumulate(y_right)
            #     y[atm_mask] = y_monotonic

        # # rolling median smoothing (works best after sorting)
        # if smooth_window > 1:
        #     y = (
        #         pd.Series(y)
        #         .rolling(window=smooth_window, center=True, min_periods=1)
        #         .median()
        #         .to_numpy(dtype=float)
        #     )

        # fit interpolator
        if use_pchip:
            sp = PchipInterpolator(x, y)  # shape-preserving
        else:
            sp = UnivariateSpline(x, y, k=3, s=spline_s, ext=3)

        splines[T_mid] = sp
        meta_rows.append({
            "T_bin": int(T_bin),
            "T_mid": T_mid,
            "n": int(x.size),
            "x_min": float(x.min()),
            "x_max": float(x.max()),
        })

    if not meta_rows:
        return splines, pd.DataFrame(columns=["T_bin", "T_mid", "n", "x_min", "x_max"])

    meta_df = pd.DataFrame(meta_rows).sort_values("T_mid").reset_index(drop=True)
    return splines, meta_df


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_surface_errors(df_raw, splines, meta_df):
    """
    Computes predicted IV for each point in df_raw using the fitted splines.
    Interpolates linearly between maturity buckets.
    """
    df = df_raw.copy()
    preds = []

    # Sort meta to ensure we can find neighbors easily
    sorted_meta = meta_df.sort_values('T_mid')
    available_Ts = sorted_meta['T_mid'].values

    for idx, row in df.iterrows():
        t_obs = row['T']
        k_obs = row['moneyness']

        # 1. Find bounding splines (Time Interpolation)
        # Find the maturity buckets just below and just above the observation
        if t_obs <= available_Ts[0]:
            # Extrapolation (or snap to first) - usually risky, but use first spline
            t_left, t_right = available_Ts[0], available_Ts[0]
        elif t_obs >= available_Ts[-1]:
            # Extrapolation (or snap to last)
            t_left, t_right = available_Ts[-1], available_Ts[-1]
        else:
            # Interpolation
            idx_right = np.searchsorted(available_Ts, t_obs)
            t_left = available_Ts[idx_right - 1]
            t_right = available_Ts[idx_right]

        # 2. Evaluate Splines at Moneyness K
        try:
            val_left = float(splines[t_left](k_obs))
            val_right = float(splines[t_right](k_obs))

            # 3. Time Interpolation
            if t_left == t_right:
                iv_pred = val_left
            else:
                # Linear weighting based on distance
                weight = (t_obs - t_left) / (t_right - t_left)
                iv_pred = val_left * (1 - weight) + val_right * weight

        except Exception:
            # Handle cases where moneyness is out of spline bounds
            iv_pred = np.nan

        preds.append(iv_pred)

    df['Predicted_IV'] = preds
    df['Error'] = df['IV'] - df['Predicted_IV']  # Signed Error (Bias)
    df['Abs_Error'] = df['Error'].abs()  # Magnitude (Accuracy)

    return df


from scipy.interpolate import griddata


def plot_smooth_error_surface(df_error):
    # Create a grid
    grid_x, grid_y = np.mgrid[
        df_error['T'].min():df_error['T'].max():100j,
        df_error['moneyness'].min():df_error['moneyness'].max():100j
    ]

    # Interpolate the ERROR values onto the grid
    grid_z = griddata(
        (df_error['T'], df_error['moneyness']),
        df_error['Error'],
        (grid_x, grid_y),
        method='linear'  # Linear is safer for errors than cubic
    )

    plt.figure(figsize=(10, 6))
    #plt.pcolormesh(grid_x, grid_y, grid_z, cmap='hot', shading='auto', vmin=-0.01, vmax=0.01)
    plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='seismic', vmin=-0.01, vmax=0.01)
    plt.colorbar(label='Error (Raw Vol Pts)')
    plt.scatter(df_error['T'], df_error['moneyness'], c='red', s=100, label='Data Points')

    plt.title('Interpolated Error Surface (Zones of Poor Fit)')
    plt.xlabel('Maturity T')
    plt.ylabel('Moneyness K/S')
    plt.legend()
    plt.show()

def eval_splined_surface(
    splines: Dict[float, object],
    meta_df: pd.DataFrame,
    m_grid: np.ndarray,
    T_grid: np.ndarray,
    min_points: int = 2,
    smooth_window: int = 4,
    clamp: Tuple[float, float] = (0.0, 5.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate splined IV surface on (T_grid, m_grid).

    1) Evaluate each maturity-slice spline only within its x-support.
    2) For each m-grid column, linearly interpolate IV across maturity (ignoring NaNs).
       Uses left/right=np.nan to avoid flat extrapolation beyond observed maturities.
    3) Optional rolling-mean smoothing along maturity for each m column.
    4) Clamp to [clamp[0], clamp[1]].

    Returns:
      IV_grid: shape (len(T_grid), len(m_grid))
      T_sorted: maturities used for slice evaluation
      IV_slices: shape (len(T_sorted), len(m_grid)) slice evaluations before T interpolation
    """
    required = {"T_mid", "x_min", "x_max"}
    missing = required - set(meta_df.columns)
    if missing:
        raise ValueError(f"meta_df is missing columns: {sorted(missing)}")

    if min_points < 2:
        raise ValueError("require_points_per_m must be >= 2")
    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1")

    m_grid = np.asarray(m_grid, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)

    # Keep only rows whose T_mid exists in splines (robust against float quirks)
    meta = meta_df.copy()
    meta["T_mid"] = meta["T_mid"].astype(float)
    meta = meta[meta["T_mid"].map(splines.__contains__)]

    if meta.empty:
        raise ValueError("No meta_df rows match spline keys.")

    meta = meta.sort_values("T_mid").reset_index(drop=True)
    T_sorted = meta["T_mid"].to_numpy(dtype=float)

    IV_slices = np.full((T_sorted.size, m_grid.size), np.nan, dtype=float)

    # --- 1) evaluate each maturity slice within its support
    # Evaluate each maturity slice only within its support
    for i, row in enumerate(meta.itertuples(index=False)):
        T_mid = float(row.T_mid)  # Ensure float

        # --- FIX: Safety Lookup ---
        if T_mid not in splines:
            # Try finding the closest key if exact match fails (float error)
            keys = np.array(list(splines.keys()))
            if len(keys) > 0:
                nearest = keys[np.argmin(np.abs(keys - T_mid))]
                if abs(nearest - T_mid) < 1e-9:  # Tolerance
                    T_mid = nearest
                else:
                    continue  # Should not happen if filtered correctly

        x_min = float(row.x_min)
        x_max = float(row.x_max)

        mask = (m_grid >= x_min) & (m_grid <= x_max)
        if mask.any():
            IV_slices[i, mask] = splines[T_mid](m_grid[mask])

    # --- 2) interpolate across maturity for each m, ignoring NaNs
    IV_grid = np.full((T_grid.size, m_grid.size), np.nan, dtype=float)

    for j in range(m_grid.size):
        col = IV_slices[:, j]
        ok = np.isfinite(col)
        if ok.sum() < min_points:
            continue

        interp_col = np.interp(
            T_grid,
            T_sorted[ok],
            col[ok],
            left=np.nan,
            right=np.nan,
        )

        # --- 3) optional smoothing along maturity dimension
        # if smooth_window > 1:
        #     interp_col = (
        #         pd.Series(interp_col)
        #         .rolling(window=smooth_window, center=True, min_periods=1)
        #         .mean()
        #         .to_numpy(dtype=float)
        #     )

        IV_grid[:, j] = interp_col

    # --- 4) clamp
    lo, hi = clamp
    IV_grid = np.clip(IV_grid, lo, hi)

    return IV_grid, T_sorted, IV_slices


def OLS_regression(ts_features, exo_df, target_maturity = 30, target_label = None, feature_columns = None):
    labels = ts_features[ts_features['maturity_days'] == target_maturity].copy()
    # filter on the target maturity
    labels = labels.set_index('date').sort_index()
    y = labels[target_label].diff().dropna()
    y.name = f"d_{target_label}"

    features = exo_df.copy()
    if feature_columns is None:
        feature_columns = features.columns

    # Normalize index immediately
    features.index = pd.to_datetime(features.index).normalize()
    # Forward fill daily signals to handle weekends/holidays before alignment
    features = features.ffill()

    # Select only the dates that match our Volatility observations
    features_aligned = features.reindex(y.index)

    x = pd.DataFrame(index=y.index)

    for col in feature_columns:
        # SPECIAL HANDLING: Do not diff Returns, only Macro levels
        if col == 'rtn':
            # Use the Level of returns (Stationary)
            vals = features_aligned[col]
            x['neg_rtn'] = np.where(vals < 0, vals, 0)
            x['pos_rtn'] = np.where(vals > 0, vals, 0)
        else:
            # Diff Macro variables (Non-Stationary Yields/Spreads)
            x[f"d_{col}"] = features_aligned[col].diff()

    # 4. Final Alignment
    # Join and drop NaNs created by the .diff() operations
    regression_df = pd.DataFrame(y).join(x, how='inner').dropna()

    y_final = regression_df[y.name]
    x_final = regression_df.drop(columns=[y.name])

    # 5. Run Regression
    x_final = sm.add_constant(x_final)
    model = sm.OLS(y_final, x_final)
    results = model.fit()

    #print(results.summary())

    return results


def snap_maturities_to_grid(df, tolerances):
    """
    tolerances = {30: 5, 45: 1, 60: 7}
    Maps maturities to targets based on specific window sizes.
    """
    df_clean = df.copy()

    def get_nearest(val):
        best_target = np.nan
        min_dist = float('inf')

        # Check each target against its specific tolerance
        for target, tol in tolerances.items():
            dist = abs(val - target)
            if dist <= tol:
                # If valid, is it the closest target?
                if dist < min_dist:
                    min_dist = dist
                    best_target = target
        return best_target

    # Apply mapping
    df_clean['maturity_days'] = df_clean['maturity_days'].apply(get_nearest)

    # Drop rows that didn't match any target
    return df_clean.dropna(subset=['maturity_days'])



