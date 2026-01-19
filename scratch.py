from main import exo_df

data = pd.read_pickle(os.path.join("data", "mydata.pkl"))
iv = ImpliedVolatility(spy, r, q)
# determine iv from black-scholes
iv_surface = iv.calculate_implied_volatility(num_expiries, data= data)
# plot a histogram of maturities
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.hist(iv_surface['T'] * 365, bins=30, color='blue', alpha=0.7)
plt.xlabel("Time to Expiration (Days)")
plt.ylabel("Frequency")
plt.xticks(np.linspace(1, 90,15))
plt.grid()
plt.show()

m,t = np.meshgrid(fivs.moneyness, fivs["T"])
points = fivs[['T', 'moneyness']].values
values = fivs['IV'].values

# Create a new, regular grid (mesh) to interpolate onto
grid_x, grid_y = np.meshgrid(np.linspace(fivs['T'].min(), fivs['T'].max(), 50),
                             np.linspace(fivs['moneyness'].min(), fivs['moneyness'].max(), 50))

# Interpolate the 'IV' values using 'cubic' method for a smooth surface
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
with plt.style.context('dark_background'):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(observation_data['T'], observation_data['moneyness'], observation_data['IV'], c='red', marker='o', alpha=0.8, label='Data Points')
    ax.plot_surface(TT, MM, IV_grid, cmap="viridis")
    ax.set_ylabel("Moneyness K/S")
    ax.set_xlabel("Maturity T (years)")
    # invert y and z by reversing current limits
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_zlim(ax.get_zlim()[::-1])
    ax.set_zlabel("Implied Vol")
    #ax.set_title("Implied Volatility Surface (Calls only)")
    #plt.savefig('figures/SPY_call_option_IV_surface.png')
    plt.show()

# multi-collinear check
corr_matrix = exo_df.corr()
print("Correlation Matrix of Exogenous Variables:")
print(corr_matrix)
# output to heatmap
with plt.style.context('dark_background'):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use contourf for a smooth heatmap representation
    ax.imshow(corr_matrix, cmap='inferno', interpolation='nearest')
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.index)
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("Moneyness K/S")
    # colorbar
    cbar = fig.colorbar(ax.imshow(corr_matrix, cmap='inferno', interpolation='nearest'))
    cbar.set_label('Correlation Coefficient')

    # ax.set_title("Implied Volatility Heatmap")
    plt.show()


# Source - https://stackoverflow.com/a
# Posted by tsvikas, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-17, License - CC BY-SA 4.0

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df_report)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc

from Utilities import prepare_for_OLS

target = 14 # 14 days in years
parameters = ['level', 'skew', 'curvature']
tolerances = {14: 5, 30: 5, 45: 5, 60: 5}  # Your "Hybrid" Strategy
labels, features = prepare_for_OLS(ts_features=ts_clean, exo_df=exo_df, target_maturity=target, target_label='level')


def animate_model_vs_market(df_actual_params, df_predicted_params, target_maturity=0.04):
    """
    Animates the evolution of the Actual vs Predicted Volatility Smile.
    """
    # 1. Align Dates
    common_dates = df_actual_params.index.intersection(df_predicted_params.index)
    actuals = df_actual_params.loc[common_dates]
    preds = df_predicted_params.loc[common_dates]

    # Moneyness Grid (80% to 120%)
    m_grid = np.linspace(0.8, 1.2, 100)
    # Log-Moneyness for the polynomial (Matches standard SVI/Quadratic fits)
    x_grid =m_grid - 1.0
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.close()  # Suppress static output

    ax.set_xlim(0.8, 1.2)

    # Dynamic Y-Limits (with some buffer)
    # We calculate global min/max to avoid jittery axis rescaling
    y_min = 0.1 #min(actuals['level'].min(), preds['level'].min()) * 0.8
    y_max = 0.4 #max(actuals['level'].max() + actuals['curvature'].max() * 0.02, preds['level'].max()) * 1.3
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel('Moneyness (K/S)')
    ax.set_ylabel('Implied Volatility')
    ax.set_title(f'Market vs Macro-Model: {target_maturity * 365:.0f} Day Volatility')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axvline(1.0, color='gray', linewidth=0.5)

    # Lines
    line_actual, = ax.plot([], [], 'b-', linewidth=3, label='Actual Market (Spline)')
    line_pred, = ax.plot([], [], 'r--', linewidth=2, label='Macro Model Prediction')
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12)
    ax.legend(loc='upper right')

    # 3. Update Function
    def update(date):
        a = actuals.loc[date]
        p = preds.loc[date]

        # Quadratic Formula: Vol = Level + Skew*x + Curvature*x^2
        vol_actual = a['level'] + a['skew'] * x_grid + a['curvature'] * (x_grid ** 2)
        vol_pred = p['level'] + p['skew'] * x_grid + p['curvature'] * (x_grid ** 2)

        line_actual.set_data(m_grid, vol_actual)
        line_pred.set_data(m_grid, vol_pred)
        time_text.set_text(f"Date: {date.strftime('%Y-%m-%d')}")
        return line_actual, line_pred, time_text

    print(f"Generating animation for {len(common_dates)} frames...")
    ani = animation.FuncAnimation(fig, update, frames=common_dates, interval=100, blit=True)

    # Save
    filename = f'Smile_Evolution_{int(target_maturity * 365)}D.mp4'
    # Use 'pillow' writer for GIFs if ffmpeg is missing
    try:
        ani.save(filename, writer='ffmpeg', fps=5)
    except:
        ani.save(filename.replace('.mp4', '.gif'), writer='pillow', fps=10)

    print(f"Saved: {filename}")
    return ani


def run_animation_pipeline(ts_features, exo_df, target_days=14):
    """
    Runs the regressions, reconstructs levels, and triggers animation.
    """
    target_mat_years = target_days / 365.0

    # 1. Clean Data using your tolerance logic
    tolerances = {14: 5, 30: 5, 45: 5, 60: 5}
    ts_clean = snap_maturities_to_grid(ts_features, tolerances)

    # 2. Containers for results
    df_actual = pd.DataFrame()
    df_pred = pd.DataFrame()

    # 3. Loop through Parameters
    for param in ['level', 'skew', 'curvature']:
        print(f"Modeling {param}...")

        # A. Run Regression (Predicts CHANGES)
        results, model = OLS_regression(ts_clean, exo_df, target_maturity=target_days, target_label=param)

        # B. Get Actual Levels (Aligned to Regression Dates)
        # OLS_regression returns a fitted model on valid dates. We need those dates.
        valid_dates = results.fittedvalues.index

        # Get actual LEVEL for t-1 (Previous Day)
        # We shift actuals by 1 to add the 'change' to yesterday's level
        actual_levels = ts_clean[ts_clean['maturity_days'] == target_days].set_index('date')[param]
        prev_actual_levels = actual_levels.shift(1).reindex(valid_dates)

        # C. Reconstruct: Pred_Level_t = Actual_Level_(t-1) + Pred_Change_t
        pred_change = results.fittedvalues
        reconstructed_level = prev_actual_levels + pred_change

        # Store
        df_pred[param] = reconstructed_level
        df_actual[param] = actual_levels.reindex(valid_dates)

    # 4. Clean up (Drop any NaN from shifting)
    df_pred.dropna(inplace=True)
    df_actual.dropna(inplace=True)

    # 5. Animate
    return df_pred #animate_model_vs_market(df_actual, df_pred, target_maturity=target_mat_years)

pred_df = run_animation_pipeline(ts_features, exo_df, target_days=14)

def plot_model_performance(mymodel, y_final):
    with plt.style.context('dark_background'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # Top Panel: Actual vs Predicted
        ax1.plot(y_final.index, y_final, label=f'Actual Level ({target_days}D)', color='white', alpha=0.6, linewidth=1.5)
        ax1.plot(y_final.index, mymodel.fittedvalues, label='Predicted Level', color='red', linestyle='--', alpha=0.8)
        ax1.set_title(f'Model Performance: {target_days}D Volatility Level', fontsize=14)
        ax1.set_ylabel('Implied Volatility')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom Panel: Residuals
        residuals = mymodel.resid
        ax2.bar(residuals.index, residuals, color='white', alpha=0.5, label='Residuals', width=1.0)
        ax2.axhline(0, color='white', linewidth=1)
        ax2.set_ylabel('Error')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    return

ts_clean = snap_maturities_to_grid(ts_features, tolerances)
target_days = 14
param = 'level'
y_final, x_final = prepare_for_OLS(ts_clean, exo_df, target_maturity=14, target_label='level')
results, model = OLS_regression(ts_clean, exo_df, target_maturity=target_days, target_label=param)
plot_model_performance(results, y_final)

import pandas as pd
import numpy as np


def create_df_reconstructed(df_raw, df_pred, target_days=14):
    """
    Creates the comparison dataframe for error analysis.

    df_raw:  The original data (Must have: 'date', 'moneyness', 'IV', 'maturity_days')
    df_pred: The regression output (Must have index='date', cols=['level', 'skew', 'curvature'])
    """
    # 1. Filter Raw Data for the specific maturity bucket (e.g., 14 days)
    # Use your tolerance logic here (e.g., 14 +/- 2 days)
    mask = (df_raw['maturity_days'] >= target_days - 2) & \
           (df_raw['maturity_days'] <= target_days + 2)
    subset_raw = df_raw[mask].copy()

    # 2. Merge Model Parameters onto the Raw Data
    # This attaches the specific day's predicted Skew/Curvature to every trade on that day
    df_reconstructed = pd.merge(
        subset_raw,
        df_pred,
        left_on='date',
        right_index=True,
        how='inner'  # Only keep days where we have both data and a prediction
    )

    # 3. Calculate the "Theoretical IV" using your Quadratic Model
    # IMPORTANT: Use (K/S - 1) for centering, matching your animation logic
    x = df_reconstructed['moneyness'] - 1.0

    df_reconstructed['Predicted_IV'] = (
            df_reconstructed['level'] +
            df_reconstructed['skew'] * x +
            df_reconstructed['curvature'] * (x ** 2)
    )

    # 4. Calculate the Error
    df_reconstructed['Error'] = df_reconstructed['IV'] - df_reconstructed['Predicted_IV']
    df_reconstructed['Abs_Error'] = df_reconstructed['Error'].abs()

    return df_reconstructed

# --- USAGE ---
# Assuming 'df_all_options' is your big dataset and 'df_pred' comes from the animation step:
# Prepare df_raw
iv_surface['maturity_days'] = (iv_surface['T'] * 365).astype(int)
df_raw = iv_surface[['date', 'moneyness', 'IV', 'maturity_days']]

# Ensure df_pred is already created using run_animation_pipeline
# Example:
# pred_df = run_animation_pipeline(ts_features, exo_df, target_days=14)

# Call create_df_reconstructed
df_reconstructed = create_df_reconstructed(df_raw, pred_df, target_days=14)

# Inspect the result
print(df_reconstructed.head())
# Now you can inspect it:

plot_prediction_accuracy(df_reconstructed)
plot_error_structure(df_reconstructed)
plot_rmse_evolution(df_reconstructed)