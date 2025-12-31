import os
import yfinance as yf
import numpy as np
import pandas as pd
from ImpliedVolatility import ImpliedVolatility
from ReadOptionData import get_option_data
from Utilities import interpolated_spline, eval_splined_surface, build_mesh, build_interpolated

spy = "SPY"
#new_spy = get_option_data("SPY")
r = 0.02 # risk free rate
q = 0.015  # 1.5% dividend yield (rough but fine)
num_expiries = 100

# get options data
#mydata = get_option_data(spy)
#mydata.to_pickle(os.path.join("data", "mydata.pkl"))
mydata = pd.read_pickle(os.path.join("data", "mydata.pkl"))

iv = ImpliedVolatility(spy, r, q)
c_pts = iv.calculate_implied_volatility(num_expiries, data= mydata)
smooth = 5
splines, meta = interpolated_spline( c_pts, T_bin_width=4/365, min_points=5, use_pchip=True, smooth_window=smooth )

m_grid, T_grid = build_mesh(c_pts, meta, n_moneyness=10, n_maturity=10)
MM, TT = np.meshgrid(m_grid, T_grid)
IV_grid, T_sorted, slice_vals = eval_splined_surface(splines, meta, m_grid, T_grid, require_points_per_m=6, smooth_window=smooth)

iv.plot_surface(IV_grid, TT, MM)
iv.plot_heatmap(IV_grid, TT, MM)

import matplotlib.pyplot as plt
with plt.style.context('dark_background'):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use contourf for a smooth heatmap representation
    cp = ax.contourf(TT, MM, IV_grid, levels=100, cmap='viridis')
    cbar = fig.colorbar(cp)
    cbar.set_label('Implied Volatility')

    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("Moneyness K/S")
    ax.set_title("Implied Volatility Heatmap")
    plt.show()