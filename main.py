import os
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import alpha
from statsmodels.graphics.correlation import plot_corr_grid

from ImpliedVolatility import ImpliedVolatility
from ReadOptionData import get_option_data
from Utilities import interpolated_spline, eval_splined_surface, build_mesh, OLS_regression, snap_maturities_to_grid
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


label_columns = 'level'
feature_columns = None

# 1. Setup
maturity_dates = [30, 45, 60]
parameters = ['level', 'skew', 'curvature']
tolerances = {30: 5, 45: 1, 60: 7}  # Your "Hybrid" Strategy

# Dictionary to hold all results
full_report_data = []

# 2. The Master Loop
for param in parameters:
    print(f"--- ANALYZING {param.upper()} ---")

    for target in maturity_dates:
        # A. Snap to grid (using hybrid tolerance)
        # We re-run this inside simply to be safe/explicit for each iteration
        ts_clean = snap_maturities_to_grid(ts_features, tolerances)

        # B. Run Regression
        # Note: 'target_label' argument allows switching between level/skew/curv
        results = OLS_regression(
            ts_clean,
            exo_df,
            target_maturity=target,
            target_label=param
        )

        # C. Extract Data for DataFrame
        # We store it in a row-based format first, then pivot later
        model_r2 = results.rsquared
        n_obs = results.nobs

        for var in results.params.index:
            coef = results.params[var]
            pval = results.pvalues[var]

            # Determine significance stars
            stars = ""
            if pval < 0.01:
                stars = "***"
            elif pval < 0.05:
                stars = "**"
            elif pval < 0.10:
                stars = "*"

            full_report_data.append({
                'Parameter': param,
                'Maturity': f"{target}D",
                'Variable': var,
                'Coefficient': f"{coef:.4f}{stars}",
                'R2': model_r2,
                'N': int(n_obs),
                'Raw_PVal': pval  # Keep raw for sorting/checking if needed
            })

# 3. Create the Master DataFrame
df_report = pd.DataFrame(full_report_data)
print(df_report)

#################################################
## DIAGNOSIS