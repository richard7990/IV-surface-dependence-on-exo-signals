import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy.interpolate import griddata
from Utilities import  build_mesh, eval_splined_surface, interpolated_spline, calculate_surface_errors, plot_smooth_error_surface

def ts_surface_features(input_df, maturity_slices, visual=False, smooth=3, n_m=50, n_t=50, min_points=5, maturity_bin_width = 4/365):
    """
    Generates a time-series of surface features (Level, Slope, Curvature)
    from a dataset of options snapshots (e.g., weekly data).

    :param input_df: DataFrame containing ['date', 'moneyness', 'T', 'IV']
    :param maturity_slices: List of target maturities in years (e.g., [30/365, 60/365])
    :param visual: Boolean, if True could optionally plot (skipped here for speed)
    :param smooth: Window size for the smoothing filter
    :return: DataFrame with columns ['date', 'maturity_days', 'level', 'skew', 'curvature']
    """

    # 1. Prepare the Groupby object (Weekly/Daily Snapshots)
    # Ensure date is datetime to avoid sorting issues
    df = input_df.copy()
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])

    grouped = df.groupby('date')
    all_features = []

    # Storage for visualization: {maturity: [(date, m_grid, iv_curve), ...]}
    visual_storage = {}
    if visual:
        visual_storage = {m: [] for m in maturity_slices}

    print(f"Processing {len(grouped)} snapshots...")

    # DEBUG: Select only the group at index 1 to test
    fobservation_date, fobservation_data = list(grouped)[1]
    for observation_date, observation_data in grouped:
        try:
            # A. Build the Surface (Fit the PCHIP Splines)
            # We use the robust settings we tuned: PCHIP + Outlier Removal + Monotonicity check
            splines, meta = interpolated_spline(
                observation_data,
                T_bin_width=maturity_bin_width,
                min_points=min_points,
                use_pchip=True,
                smooth_window=smooth,
            )

            # Skip if data was too sparse to generate any splines
            if not splines:
                continue

            # B. Build the Grid (Dense Mesh for Evaluation)
            # This generates the consistent grid needed for slope/curvature calcs
            m_grid, T_grid = build_mesh(observation_data, meta, n_moneyness=n_m, n_maturity=n_t)

            # C. Evaluate the Surface (Fill the Grid)
            # This applies the smoothing and gap-filling logic to generate the full surface
            IV_grid, T_sorted, slice_vals = eval_splined_surface(
                splines, meta, m_grid, T_grid,
                min_points=min_points,
                smooth_window=smooth
            )

            if np.isnan(IV_grid).all():
                print(f"Skipping date {observation_date.date()} due to empty IV grid.")
                continue

            # D. Extract Features (Level, Slope, Curvature)
            # Uses the time-interpolated extraction to prevent "staircase" jumps
            daily_feats = get_surface_features(IV_grid, m_grid, T_grid, maturity_slices)

            # Add the date column so we know when this observation happened
            daily_feats['date'] = observation_date

            # debug pause
            if observation_date == fobservation_date:
                print(f"Processed date: {observation_date.date()}")
                with plt.style.context('dark_background'):
                    MM, TT = np.meshgrid(m_grid, T_grid)

                    # Usage:
                    df_error = calculate_surface_errors(observation_data, splines, meta)
                    plot_smooth_error_surface(df_error)

                    fig = plt.figure(figsize=(10, 6))
                    ax = fig.add_subplot(111, projection="3d")
                    ax.scatter(observation_data['T'], observation_data['moneyness'], observation_data['IV'], c='red',
                               marker='o', alpha=0.8, label='Data Points')
                    #ax.plot_surface(TT, MM, Z, cmap="viridis")
                    ax.plot_surface(TT, MM, IV_grid, cmap="inferno")
                    ax.set_ylabel("Moneyness K/S")
                    ax.set_xlabel("Maturity T (years)")
                    # invert y and z by reversing current limits
                    ax.set_xlim(ax.get_xlim()[::-1])
                    ax.set_zlim(ax.get_zlim()[::-1])
                    ax.set_zlabel("Implied Vol")
                    # ax.set_title("Implied Volatility Surface (Calls only)")
                    # plt.savefig('figures/SPY_call_option_IV_surface.png')
                    plt.show()

            if visual:
                # Extract the full IV curve for each target maturity for plotting
                for target_T in maturity_slices:
                    # Clamp target_T to grid bounds
                    t_clamp = max(T_grid[0], min(target_T, T_grid[-1]))

                    # Find indices for interpolation
                    idx = np.searchsorted(T_grid, t_clamp)

                    if idx == 0:
                        iv_curve = IV_grid[0, :]
                    elif idx >= len(T_grid):
                        iv_curve = IV_grid[-1, :]
                    else:
                        T0 = T_grid[idx - 1]
                        T1 = T_grid[idx]
                        w = (t_clamp - T0) / (T1 - T0)
                        iv_curve = (1 - w) * IV_grid[idx - 1, :] + w * IV_grid[idx, :]

                    visual_storage[target_T].append((observation_date, m_grid, iv_curve))

            all_features.append(daily_feats)

        except Exception as e:
            # Catch errors for a single bad week so the whole loop doesn't crash
            print(f"Skipping date {observation_date.date()} due to error: {e}")
            continue

    # 2. Visualization (if requested)
    if visual and visual_storage:
        for target_T, curves in visual_storage.items():
            if not curves:
                continue
            with plt.style.context('dark_background'):
                    fig, ax = plt.subplots(figsize=(10, 5))

                    N_curves = len(curves)

                    # 1. Choose a colormap (e.g., 'viridis', 'plasma', 'coolwarm')
                    cmap = plt.get_cmap('inferno')

                    for i, (d, m, iv) in enumerate(curves):
                        # 2. Calculate the fraction of time (0.0 to 1.0)
                        fraction = i / max(N_curves - 1, 1)

                        # 3. Get the specific color for this time step
                        curve_color = cmap(fraction)

                        # Plot with the dynamic color (keep alpha high to see colors clearly)
                        ax.plot(m, iv, color=curve_color, alpha=0.8, linewidth=1.2)

                    ax.axvline(1.0, color='w', linestyle='--', alpha=0.5, label="ATM")

                    # 4. Create a Colorbar to act as the Legend
                    # We create a "ScalarMappable" just to generate the colorbar
                    norm = mcolors.Normalize(vmin=0, vmax=N_curves)
                    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])

                    # Add the colorbar to the plot
                    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
                    cbar.set_label('Time Progression (Older -> Newer)', rotation=270, labelpad=15)
                    # Optional: Replace ticks with "Start" and "End" if you prefer
                    cbar.set_ticks([0, N_curves])
                    cbar.set_ticklabels(['Start Date', 'End Date'])

                    ax.set_title(
                        f"Volatility Smile Evolution: T={target_T:.3f} yrs ({target_T * 365:.1f} days)")
                    ax.set_xlabel("Moneyness (K/S)")
                    ax.set_ylabel("Implied Vol")
                    ax.grid(True, alpha=0.3)
                    plt.savefig(f"figures/SPY_{target_T * 365}_day SMILE evolution.png")
                    plt.show()

    # 3. Compile into a single DataFrame
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)

        # Organize columns nicely
        cols = ['date', 'maturity_days', 'level', 'skew', 'curvature']
        # Add any other cols that might exist (like maturity_years)
        remaining_cols = [c for c in final_df.columns if c not in cols]
        final_df = final_df[cols + remaining_cols]

        return final_df.sort_values(['date', 'maturity_days'])
    else:
        return pd.DataFrame()


def get_surface_features(IV_grid, m_grid, T_grid, maturity_slices):
    """
    Extracts Level, Skew, and Curvature from the finalized IV_grid.
    Uses linear interpolation between time steps to avoid jumps.
    """
    maturity_slices = np.asarray(maturity_slices, dtype=float)
    results = []

    # Pre-calculate gradients for the whole grid to match your smoothing logic
    # (Doing this once is faster and consistent)
    # Axis 1 is Moneyness (along rows)
    skew_grid = np.gradient(IV_grid, m_grid, axis=1)
    curv_grid = np.gradient(skew_grid, m_grid, axis=1)

    # Find ATM index once
    # Ensure m_grid is sorted or this argmin works as expected (it usually does)
    atm_idx = np.argmin(np.abs(m_grid - 1.0))

    for target_T in maturity_slices:
        # 1. Handle Out of Bounds
        if target_T < T_grid[0]:
            target_T = T_grid[0]
        if target_T > T_grid[-1]:
            target_T = T_grid[-1]

        # 2. Find Bracketing Indices (Time Interpolation)
        # searchsorted finds the first index where T_grid[i] >= target_T
        idx = np.searchsorted(T_grid, target_T)

        if T_grid[idx] == target_T:
            # Exact match
            level = IV_grid[idx, atm_idx]
            skew = skew_grid[idx, atm_idx]
            curv = curv_grid[idx, atm_idx]
        else:
            # Interpolate between idx-1 and idx
            prev_idx = max(0, idx - 1)

            T0 = T_grid[prev_idx]
            T1 = T_grid[idx]

            # Weight for linear interpolation
            w = (target_T - T0) / (T1 - T0)

            # Interpolate the ATM values directly
            # Level
            L0 = IV_grid[prev_idx, atm_idx]
            L1 = IV_grid[idx, atm_idx]
            level = (1 - w) * L0 + w * L1

            # Skew
            S0 = skew_grid[prev_idx, atm_idx]
            S1 = skew_grid[idx, atm_idx]
            skew = (1 - w) * S0 + w * S1

            # Curvature
            C0 = curv_grid[prev_idx, atm_idx]
            C1 = curv_grid[idx, atm_idx]
            curv = (1 - w) * C0 + w * C1

        results.append([target_T, level, skew, curv])

    # Convert to DataFrame
    df = pd.DataFrame(results, columns=["maturity_years", "level", "skew", "curvature"])
    df["maturity_days"] = (df["maturity_years"] * 365).astype(int)

    return df