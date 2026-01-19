import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Utilities import OLS_regression, snap_maturities_to_grid


def process_results(ts_features, exo_df, target_days=14):
    """
    Runs OLS regressions for Level, Skew, and Curvature for a specific target maturity.

    Returns:
        df_pred (pd.DataFrame): Reconstructed time series (Actual vs Predicted) for animation.
        df_report (pd.DataFrame): Statistical report (Coefs, R2, P-values) for this maturity.
    """

    # 1. Setup & Cleaning
    tolerances = {14: 5, 30: 5, 45: 5, 60: 5}  # "Hybrid" tolerance strategy
    ts_clean = snap_maturities_to_grid(ts_features, tolerances)

    # Containers
    df_actual = pd.DataFrame()
    df_pred = pd.DataFrame()
    report_rows = []

    parameters = ['level', 'skew', 'curvature']

    print(f"--- Processing Maturity: {target_days}D ---")

    # 2. Loop through Parameters (Level, Skew, Curvature)
    for param in parameters:
        # ---------------------------------------------------------
        # A. Run Regression
        # ---------------------------------------------------------
        # Assumes OLS_regression returns (results_wrapper, model_object)
        results = OLS_regression(ts_clean, exo_df, target_maturity=target_days, target_label=param)

        # ---------------------------------------------------------
        # B. Build Statistical Report
        # ---------------------------------------------------------
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

            report_rows.append({
                'Maturity': f"{target_days}D",
                'Parameter': param.capitalize(),
                'Variable': var,
                'Coefficient': f"{coef:.4f}{stars}",
                'R2': f"{model_r2:.4f}",
                'N': int(n_obs),
                'Raw_PVal': pval
            })

        # ---------------------------------------------------------
        # C. Reconstruct Predictions (Level_t = Level_{t-1} + Predicted_Change_t)
        # ---------------------------------------------------------
        valid_dates = results.fittedvalues.index

        # Get actual LEVEL for t-1 (Previous Day)
        # We need the previous day's actual value to add the predicted change to
        actual_levels_series = ts_clean[ts_clean['maturity_days'] == target_days].set_index('date')[param]

        # Shift actuals by 1 to align t-1 with t
        prev_actual_levels = actual_levels_series.shift(1).reindex(valid_dates)

        # Reconstruct: Pred_Level = Actual_(t-1) + Pred_Change
        pred_change = results.fittedvalues
        reconstructed_level = prev_actual_levels + pred_change

        # Store in columns named by parameter
        df_pred[f"{param}_pred"] = reconstructed_level
        df_pred[f"{param}_actual"] = actual_levels_series.reindex(valid_dates)

    # 3. Finalize Outputs
    df_pred.dropna(inplace=True)
    df_report = pd.DataFrame(report_rows)

    # Optional: Set index of df_report for cleaner looking initial print
    if not df_report.empty:
        df_report.set_index(['Maturity', 'Parameter', 'Variable'], inplace=True)

    return df_pred, df_report


# Post processing scripts

def plot_prediction_accuracy(df_reconstructed, param=14):
    with plt.style.context('dark_background'):
        plt.figure(figsize=(8, 8))

        plt.scatter(
            df_reconstructed['IV'],
            df_reconstructed['Predicted_IV'],
            alpha=0.2, color='red', s=10
        )

        # 45-degree line (Perfect prediction)
        max_val = max(df_reconstructed['IV'].max(), df_reconstructed['Predicted_IV'].max())
        min_val = min(df_reconstructed['IV'].min(), df_reconstructed['Predicted_IV'].min())
        plt.plot([min_val, max_val], [min_val, max_val], '--w', linewidth=2, label='Perfect Fit')

        #plt.title('Goodness of Fit: Actual vs Predicted Implied Volatility')
        plt.xlabel('Actual Market IV')
        plt.ylabel('Macro-Model Predicted IV')
        plt.legend()
        #plt.grid(True)
        plt.savefig(f"figures/prediction_accuracy_for_IV_at_T={param}.png")
        plt.show()
    return

def plot_error_structure(df_reconstructed, param=14):
    with plt.style.context('dark_background'):
        plt.figure(figsize=(10, 6))

        # Scatter plot of ALL errors vs Moneyness
        plt.scatter(
            df_reconstructed['moneyness'],
            df_reconstructed['Error'],
            alpha=0.2, s=10, color='red', label='Individual Quote Errors'
        )

        # Add a smoothing line to show the average bias
        # (Optional: requires seaborn, but simple binning works too)
        import seaborn as sns
        sns.regplot(
            x=df_reconstructed['moneyness'],
            y=df_reconstructed['Error'],
            scatter=False, lowess=True, color='red', line_kws={'label': 'Mean Bias'}
        )

        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('Moneyness (K/S)')
        plt.ylabel('Prediction Error (Actual - Predicted)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"figures/SMILE curve error structure for IV at T={param}.png")
        plt.show()
    return

def plot_rmse_evolution(df_reconstructed, target_label='14D'):
    # Group by date to get daily error metrics
    daily_error = df_reconstructed.groupby('date').agg(
        RMSE=('Error', lambda x: np.sqrt((x ** 2).mean())),
        MAE=('Abs_Error', 'mean')
    )
    with plt.style.context('dark_background'):
        plt.figure(figsize=(10, 5))

        # Plot RMSE (Root Mean Square Error)
        plt.plot(daily_error.index, daily_error['RMSE'], color='red', label='Daily RMSE', linewidth=1.5)

        # Add context
        plt.axhline(daily_error['RMSE'].mean(), color='white', linestyle='--',
                    label=f"Avg RMSE: {daily_error['RMSE'].mean():.2f}")

        plt.ylabel('Volatility Points (RMSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"figures/rmse_evolution for {target_label}.png")
        plt.show()
    return

def plot_model_performance(df_reconstructed, param='level', maturity=14):
    """
    Plots Actual vs Predicted Level and the corresponding Residuals.

    Args:
        df_reconstructed (pd.DataFrame): DF containing '{param}_actual' and '{param}_pred'
        param (str): The parameter name (e.g., 'level', 'skew', 'curvature')
        maturity (int): The maturity days (for labeling)
    """

    # Extract series dynamically based on parameter name
    y_actual = df_reconstructed[f'{param}_actual']
    y_pred = df_reconstructed[f'{param}_pred']
    residuals = y_actual - y_pred

    # Setup the plot
    with plt.style.context('dark_background'):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Top Panel: Actual vs Predicted
        ax1.plot(y_actual.index, y_actual, label=f'Actual {param.capitalize()} ({maturity}D)',
                 color='blue', alpha=0.6, linewidth=1.5)
        ax1.plot(y_pred.index, y_pred, label=f'Predicted {param.capitalize()}',
                 color='red', linestyle='--', alpha=0.8)

        ax1.set_ylabel('Implied Volatility / Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom Panel: Residuals
        ax2.bar(residuals.index, residuals, color='gray', alpha=0.5, label='Residuals', width=1.0)
        ax2.axhline(0, color='white', linewidth=1)
        ax2.set_ylabel('Error')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.savefig(f"figures/")

        plt.tight_layout()
        plt.show()
    return

def plot_model_stability(df_reconstructed, param=14, window=50):
    """
    Calculates and plots the Rolling R-Squared (via Correlation^2)
    between Actual and Predicted values to observe model stability over time.

    Args:
        df_reconstructed (pd.DataFrame): DF containing '{param}_actual' and '{param}_pred'
        param (str): The parameter name to analyze
        window (int): Rolling window size (default 50 days)
    """

    col_actual = f'{param}_actual'
    col_pred = f'{param}_pred'

    # Calculate Rolling Correlation and Square it to get R2
    # This approximates how well the Predicted movement explains Actual movement in the window
    rolling_corr = df_reconstructed[col_actual].rolling(window=window).corr(df_reconstructed[col_pred])
    rolling_r2 = rolling_corr ** 2

    # Plot
    with plt.style.context('dark_background'):
        plt.figure(figsize=(12, 5))
        plt.plot(rolling_r2.index, rolling_r2, color='darkgreen', linewidth=2)

        plt.title(f'Rolling {window}-Day R-Squared (Model Stability): {param}', fontsize=14)
        plt.ylabel('R-Squared (Corr^2)')

        # Add mean reference line
        avg_r2 = rolling_r2.mean()
        plt.axhline(avg_r2, color='gray', linestyle='--', label=f'Average R2 ({avg_r2:.2f})')

        plt.legend()
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"figures/model_stability at {param}.png")
        plt.show()