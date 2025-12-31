import numpy as np
import pandas as pd
def get_surface_features(IV_grid, m_grid, T_grid, maturity_slices):
    maturity_slices = np.asarray(maturity_slices, dtype=float)

    # True nearest indices in T_grid
    idx_right = np.searchsorted(T_grid, maturity_slices, side="left")
    idx_left = np.clip(idx_right - 1, 0, len(T_grid) - 1)
    idx_right = np.clip(idx_right, 0, len(T_grid) - 1)

    choose_right = (np.abs(T_grid[idx_right] - maturity_slices) < np.abs(T_grid[idx_left] - maturity_slices))
    closest_indices = np.where(choose_right, idx_right, idx_left)

    # Extract slices (N_ms, M)
    IV_at_T = IV_grid[closest_indices, :]

    atm_idx = np.argmin(np.abs(m_grid - 1.0))
    level = IV_at_T[:, atm_idx]
    skew_full = np.gradient(IV_at_T, m_grid, axis=1)
    skew = skew_full[:, atm_idx]
    curv_full = np.gradient(skew_full, m_grid, axis=1)
    curvature = curv_full[:, atm_idx]

    rs = pd.DataFrame({
        "maturity_days": (maturity_slices * 365).astype(int),
        "maturity_years": maturity_slices,
        "level": level,
        "skew": skew,
        "curvature": curvature,
    })
    return rs