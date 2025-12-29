import numpy as np
from scipy.interpolate import griddata
def build_interpolated(df, MM, TT, smooth_window=5, min_pts=3, T_bin_width=7 / 365):
    """
    Robust surface builder:
      1) Bin maturities into buckets (e.g., weekly)
      2) Smooth IV across moneyness within each bucket using rolling mean
      3) Interpolate on (moneyness, T) grid using linear griddata
      4) Fill holes with nearest-neighbour

    df must have columns: ["moneyness", "T", "IV"]
    MM, TT are meshgrids of moneyness and T.
    """
    df = df.copy()
    df = df.dropna(subset=["moneyness", "T", "IV"])

    # Bin maturities into buckets
    df["T_bin"] = (df["T"] / T_bin_width).round().astype(int)

    # Sort so rolling operates in moneyness order within each maturity bin
    df = df.sort_values(["T_bin", "moneyness"])

    # Smooth across moneyness inside each T-bin
    df["IV_smooth"] = (
        df.groupby("T_bin", group_keys=False)["IV"]
        .apply(lambda s: s.rolling(window=smooth_window, center=True, min_periods=min_pts).mean())
    )

    # Use smoothed values where available
    df["IV_use"] = df["IV_smooth"].fillna(df["IV"])

    # Interpolate (linear)
    IV_grid = griddata(
        points=(df["moneyness"].values, df["T"].values),
        values=df["IV_use"].values,
        xi=(MM, TT),
        method="linear"
    )

    # Fill holes with nearest neighbour
    IV_grid_nn = griddata(
        points=(df["moneyness"].values, df["T"].values),
        values=df["IV_use"].values,
        xi=(MM, TT),
        method="nearest"
    )

    return np.where(np.isnan(IV_grid), IV_grid_nn, IV_grid)