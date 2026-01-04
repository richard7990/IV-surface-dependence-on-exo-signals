import numpy as np
import pandas as pd

def get_surface_features(IV_grid, m_grid, T_grid, maturity_slices):
    """
    Extracts Level, Skew, and Curvature from the finalized IV_grid.
    Uses linear interpolation between time steps to avoid jumps.
    """
    maturity_slices = np.asarray(maturity_slices, dtype=float)
    results = []

    # Pre-calculate gradients for the whole grid to match your smoothing logic
    # (Doing this once is faster and consistent)
    # Axis 1 is Moneyness (along rows)
    skew_grid = np.gradient(IV_grid, m_grid, axis=1)
    curv_grid = np.gradient(skew_grid, m_grid, axis=1)

    # Find ATM index once
    # Ensure m_grid is sorted or this argmin works as expected (it usually does)
    atm_idx = np.argmin(np.abs(m_grid - 1.0))

    for target_T in maturity_slices:
        # 1. Handle Out of Bounds
        if target_T < T_grid[0]:
            target_T = T_grid[0]
        if target_T > T_grid[-1]:
            target_T = T_grid[-1]

        # 2. Find Bracketing Indices (Time Interpolation)
        # searchsorted finds the first index where T_grid[i] >= target_T
        idx = np.searchsorted(T_grid, target_T)

        if T_grid[idx] == target_T:
            # Exact match
            level = IV_grid[idx, atm_idx]
            skew = skew_grid[idx, atm_idx]
            curv = curv_grid[idx, atm_idx]
        else:
            # Interpolate between idx-1 and idx
            prev_idx = max(0, idx - 1)

            T0 = T_grid[prev_idx]
            T1 = T_grid[idx]

            # Weight for linear interpolation
            w = (target_T - T0) / (T1 - T0)

            # Interpolate the ATM values directly
            # Level
            L0 = IV_grid[prev_idx, atm_idx]
            L1 = IV_grid[idx, atm_idx]
            level = (1 - w) * L0 + w * L1

            # Skew
            S0 = skew_grid[prev_idx, atm_idx]
            S1 = skew_grid[idx, atm_idx]
            skew = (1 - w) * S0 + w * S1

            # Curvature
            C0 = curv_grid[prev_idx, atm_idx]
            C1 = curv_grid[idx, atm_idx]
            curv = (1 - w) * C0 + w * C1

        results.append([target_T, level, skew, curv])

    # Convert to DataFrame
    df = pd.DataFrame(results, columns=["maturity_years", "level", "skew", "curvature"])
    df["maturity_days"] = (df["maturity_years"] * 365).astype(int)

    return df