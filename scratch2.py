import os
import numpy as np
import pandas as pd
from ImpliedVolatility import ImpliedVolatility
from ReadOptionData import get_option_data
from Utilities import interpolated_spline, eval_splined_surface, build_mesh, OLS_regression, snap_maturities_to_grid
from ExogenousSignals import get_exo_df
from SurfaceFeatures import ts_surface_features

spy = "SPY"
r = 0.02 # risk free rate
q = 0.015  # 1.5% dividend yield
num_expiries = 100 # no. of expiries to evaluate

data = pd.read_pickle(os.path.join("data", "mydata.pkl"))

iv = ImpliedVolatility(spy, r, q)
# determine iv from black-scholes
iv_surface = iv.calculate_implied_volatility(num_expiries, data= data)

# ts - IV surface spline parameters
smooth_window = 5 # smoothing window size
min_interp_points = 3 # minimum number of points for spline interpolation
maturity_bin_width = 4/365 # width of bin for maturities for grouping
n_moneyness = 100 # No. of moneyness points
n_maturity = 100 # no. of maturity points
maturity_slices = [14/365, 30/365, 45/365, 60/365] # evaluate at fixed maturity slices


# generate iv-surface and parametrise into level, skew, and curvature at fixed maturities
ts_features = ts_surface_features(iv_surface, maturity_slices = maturity_slices,
                                  visual=True, smooth=smooth_window,
                                  n_m=n_moneyness, n_t=n_maturity,
                                  min_points=min_interp_points,
                                  maturity_bin_width=maturity_bin_width)
