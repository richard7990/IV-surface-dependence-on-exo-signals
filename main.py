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

# IV surface spline parameters
smooth_window = 5 # smoothing window size
min_interp_points = 5 # minimum number of points for spline interpolation
maturity_bin_width = 4/365 # width of bin for maturities for grouping
n_moneyness = 100 # No. of moneyness points
n_maturity = 100 # no. of maturity points

# get options data
data = get_option_data(spy)
#data.to_pickle(os.path.join("data", "data.pkl"))
data = pd.read_pickle(os.path.join("data", "mydata.pkl"))

iv = ImpliedVolatility(spy, r, q)
# determine iv from black-scholes
iv_surface = iv.calculate_implied_volatility(num_expiries, data= data)
# prepare splines over the surface
splines, spline_meta_data = interpolated_spline(iv_surface,
                                    T_bin_width=maturity_bin_width,
                                    min_points=min_interp_points,
                                    use_pchip=True,
                                    smooth_window=smooth_window)

m_grid, T_grid = build_mesh(iv_surface, spline_meta_data, n_moneyness=n_moneyness, n_maturity=n_maturity)
MM, TT = np.meshgrid(m_grid, T_grid)
# evaluate the splines on the newly discretized mesh
IV_grid, T_sorted, slice_vals = eval_splined_surface(splines, spline_meta_data,
                                                     m_grid=m_grid, T_grid =T_grid,
                                                     min_points=min_interp_points,
                                                     smooth_window=smooth_window)
# plot the 3D IV surface
iv.plot_surface(IV_grid, TT, MM)
# plot the IV surface heatmap
iv.plot_heatmap(IV_grid, TT, MM)

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
plt.close('all')

# Get time series of exogenous signals
start_date = ts_features['date'].min()
end_date = ts_features['date'].max()
exo_df = get_exo_df(start_date=start_date, end_date=end_date)

label_columns = 'level'
feature_columns = None

# 1. Setup
maturity_dates = [14, 30, 45, 60]
parameters = ['level', 'skew', 'curvature']
tolerances = {14: 5, 30: 5, 45: 5, 60: 5}  # Your "Hybrid" Strategy

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
import matplotlib.pyplot as plt

#################################################
## DIAGNOSIS

mymodel = sm.OLS(y_final, x_final).fit()

# plot of actual vs the predicted level
# 2. Setup the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Top Panel: Actual vs Predicted
ax1.plot(y_final.index, y_final, label='Actual Level (30D)', color='black', alpha=0.6, linewidth=1.5)
ax1.plot(y_final.index, mymodel.fittedvalues, label='Predicted Level', color='red', linestyle='--', alpha=0.8)
ax1.set_title('Model Performance: 30D Volatility Level', fontsize=14)
ax1.set_ylabel('Implied Volatility')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom Panel: Residuals
residuals = mymodel.resid
ax2.bar(residuals.index, residuals, color='gray', alpha=0.5, label='Residuals', width=1.0)
ax2.axhline(0, color='black', linewidth=1)
ax2.set_ylabel('Error')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

## Rolling regression plot
from statsmodels.regression.rolling import RollingOLS

# 1. Define Window (e.g., 6 months = approx 126 trading days)
window_size = 50 # Short window since your sample size is small (N=73/91)

# 2. Fit Rolling Model
# Ensure your index is a datetime index for the plot to look right
rolling_model = RollingOLS(y_final, x_final, window=window_size)
rolling_res = rolling_model.fit()

# 3. Plot
plt.figure(figsize=(12, 5))
plt.plot(rolling_res.rsquared.index, rolling_res.rsquared, color='darkgreen', linewidth=2)
plt.title(f'Rolling {window_size}-Day R-Squared (Model Stability)', fontsize=14)
plt.ylabel('R-Squared')
plt.axhline(0.58, color='gray', linestyle='--', label='Full Sample R2 (0.58)') # Reference line
plt.legend()
plt.ylim(0, 1) # R2 is between 0 and 1
plt.grid(True)
plt.show()