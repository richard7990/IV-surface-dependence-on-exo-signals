import os
import numpy as np
import pandas as pd
from ImpliedVolatility import ImpliedVolatility
from ExogenousSignals import get_exo_df
from SurfaceFeatures import ts_surface_features
from ProcessResults import *

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

# Get time series of exogenous signals
start_date = ts_features['date'].min()
end_date = ts_features['date'].max()
exo_df = get_exo_df(start_date=start_date, end_date=end_date)

results_storage = {}

# 3. Master Loop
for mat in maturity_slices:
    # Run the pipeline
    df_pred, df_stats = process_results(ts_features, exo_df, target_days=mat * 365)

    # Store results
    results_storage[mat] = {
        'preds': df_pred,
        'stats': df_stats
    }
    # Optional: Print a quick summary to know it worked
    print(
        f"Stored results for {mat}D. (R2 Level: {df_stats.loc[pd.IndexSlice[:, 'Level', 'd_credit_spread'], 'R2'].values[0] if not df_stats.empty else 'N/A'})")


