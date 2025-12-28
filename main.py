import yfinance as yf
from ImpliedVolatility import ImpliedVolatility

spy = yf.Ticker("SPY")
r = 0.02 # risk free rate
q = 0.015  # 1.5% dividend yield (rough but fine)
num_expiries = 100

iv = ImpliedVolatility(spy, r, q)
tt = iv.calculate_implied_volatility(num_expiries)
iv.plot_surface(n_moneyness=30, n_maturity=30)

n_moneyness=30
n_maturity=30
c_pts = tt
import numpy as np
import pandas as pd
from datetime import date
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata, RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def interp_and_fill_binned(df, MM, TT, smooth_window=5, min_pts=3, T_bin_width=7 / 365):
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

c_pts = c_pts[(c_pts["T"] > 0.05)]
min_moneyness, max_moneyness =  c_pts["moneyness"].min(),  c_pts["moneyness"].max()
min_maturity, max_maturity =  c_pts["T"].min(),  c_pts["T"].max()

m_grid = np.linspace(min_moneyness, max_moneyness, n_moneyness)
T_grid = np.linspace(min_maturity, max_maturity, n_maturity)
MM, TT = np.meshgrid(m_grid, T_grid)
# interpolate
IV_grid = interp_and_fill_binned( c_pts, MM, TT)

interp = RegularGridInterpolator(
    (T_grid, m_grid),
    IV_grid,
    bounds_error=False,
    fill_value=np.nan
)


def IV(m, T):
    return float(interp([[T, m]])[0])

with plt.style.context('dark_background'):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # maturity slices (mask NaNs so plot doesn't break on gaps)
    for T0 in [30 / 365, 90 / 365]:
        m_line = np.linspace(m_grid.min(), m_grid.max(), 100)
        IV_line = np.array([IV(m, T0) for m in m_line])
        mask = np.isfinite(IV_line)
        if mask.any():
            ax.plot(m_line[mask], np.full_like(m_line[mask], T0), IV_line[mask] - 0.005,
                    color='black', linestyle='--', linewidth=2)

    # moneyness reference lines
    for m0 in [1.00, 1.03, 1.06]:
        T_line = np.linspace(T_grid.min(), T_grid.max(), 100)
        IV_line = np.array([IV(m0, T) for T in T_line])
        mask = np.isfinite(IV_line)
        if mask.any():
            ax.plot(np.full_like(T_line[mask], m0), T_line[mask], IV_line[mask]-0.05,
                    color='black', linestyle='--', linewidth=2)
    # draw surface slightly transparent so lines are visible
    ax.plot_surface(MM, TT, IV_grid, cmap="viridis", linewidth=0, antialiased=False, alpha=0.75)

    ax.set_xlabel("Moneyness K/S")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("Implied Vol")
    ax.set_title("IV Surface with Parameter Extraction Paths")

    plt.show()
