from Utilities import interpolated_spline, eval_splined_surface, build_mesh
import numpy as np
import pandas as pd
from SurfaceFeatures import get_surface_features
import matplotlib.pyplot as plt

def ts_surface_features(input_df, maturity_slices, visual=False, smooth=3, n_m=50, n_t=50):
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
    # observation_date, observation_data = list(grouped)[1]
    for observation_date, observation_data in grouped:
        try:
            # A. Build the Surface (Fit the PCHIP Splines)
            # We use the robust settings we tuned: PCHIP + Outlier Removal + Monotonicity check
            splines, meta = interpolated_spline(
                observation_data,
                T_bin_width=4/365,   # 4 days bin width is good for weekly data
                min_points=3,
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
                require_points_per_m=3
            )

            if np.isnan(IV_grid).all():
                print(f"Skipping date {observation_date.date()} due to empty IV grid.")
                continue

            # D. Extract Features (Level, Slope, Curvature)
            # Uses the time-interpolated extraction to prevent "staircase" jumps
            daily_feats = get_surface_features(IV_grid, m_grid, T_grid, maturity_slices)

            # Add the date column so we know when this observation happened
            daily_feats['date'] = observation_date

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
                        T0 = T_grid[idx-1]
                        T1 = T_grid[idx]
                        w = (t_clamp - T0) / (T1 - T0)
                        iv_curve = (1 - w) * IV_grid[idx-1, :] + w * IV_grid[idx, :]
                    
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
            
            plt.figure(figsize=(10, 5))
            N_curves = len(curves)
            for i, (d, m, iv) in enumerate(curves):
                # Fade from light blue to dark blue over time
                plt.plot(m, iv, color='b', alpha=(i + 1) / N_curves)
            
            plt.axvline(1.0, color='r', linestyle='--', label="ATM")
            plt.title(f"Volatility Smile Evolution: T={target_T:.3f} yrs ({target_T*365:.3f} days)")
            plt.xlabel("Moneyness")
            plt.ylabel("Implied Vol")
            plt.grid(True)
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


