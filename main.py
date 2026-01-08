import os
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import alpha
from ImpliedVolatility import ImpliedVolatility
from ReadOptionData import get_option_data
from Utilities import interpolated_spline, eval_splined_surface, build_mesh
import matplotlib.pyplot as plt
from ExogenousSignals import get_exo_df
from SurfaceFeatures import ts_surface_features

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

# Get time series of IV surface features
ts_features = ts_surface_features(c_pts, maturity_slices, visual=False)

# Get time series of exogenous signals
start_date = ts_features['date'].min()
end_date = ts_features['date'].max()
exo_df = get_exo_df(start_date=start_date, end_date=end_date)





# Generate the X values for the regression initially start with 30D maturity
Y = ts_features.copy()
target_maturity = 31 # or 31, depending on your slice input
Y =  Y[Y['maturity_days'] == 31]

Y = Y.set_index('date').sort_index()
Y['d_lvl'] = Y['level'].diff()

X = pd.DataFrame()
# resample to weekly
X_weekly = exo_df.copy().ffill()
X_weekly = X_weekly.reindex(Y.index)
# difference
X['dy10'] = X_weekly['yld_10Y'].diff()
X['dcs'] = X_weekly['credit_spread'].diff()
X['d10y2y'] = X_weekly['diff_10Y_2Y'].diff()
# implement market asymmetry (panic fear)
X['neg_rtn'] = np.where(X_weekly['rtn'] < 0, X_weekly['rtn'], 0)
X['pos_rtn'] = np.where(X_weekly['rtn'] > 0, X_weekly['rtn'], 0)


Y.index = pd.to_datetime(Y.index).normalize().tz_localize(None)
X.index = pd.to_datetime(X.index).normalize().tz_localize(None)
regression_df = Y[['d_lvl']].join(X, how='inner')
regression_df = regression_df.dropna()

# 4. Run Regression
target = regression_df['d_lvl']
features = regression_df[['dy10', 'dcs', 'd10y2y', 'neg_rtn', 'pos_rtn']]

import statsmodels.api as sm

features = sm.add_constant(features)
model = sm.OLS(target, features)
results = model.fit()

print(results.summary())