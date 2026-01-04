import os
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import alpha

from ImpliedVolatility import ImpliedVolatility
from ReadOptionData import get_option_data
from Utilities import interpolated_spline, eval_splined_surface, build_mesh, build_interpolated
import matplotlib.pyplot as plt
from SurfaceFeatures import get_surface_features
from DailyPipeline import ts_surface_features

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

m_grid, T_grid = build_mesh(c_pts, meta, n_moneyness=100, n_maturity=100)
MM, TT = np.meshgrid(m_grid, T_grid)
IV_grid, T_sorted, slice_vals = eval_splined_surface(splines, meta, m_grid, T_grid, require_points_per_m=6, smooth_window=smooth)

iv.plot_surface(IV_grid, TT, MM)
iv.plot_heatmap(IV_grid, TT, MM)

maturity_slices = [30/365, 45/365, 60/365]
my_features = get_surface_features(IV_grid, m_grid, T_grid, maturity_slices)

ts_features = ts_surface_features(c_pts, maturity_slices, visual=True)

# Assuming 'ts_results' is your final DataFrame from the previous step

plt.figure(figsize=(12, 6))
plt.plot(ts_features['date'], ts_features['level'], label='30-Day ATM Vol')
plt.title("Evolution of Volatility Level (2019)")
plt.ylabel("Implied Vol")
plt.legend()
plt.show()