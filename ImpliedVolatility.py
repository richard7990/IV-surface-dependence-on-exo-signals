import numpy as np
import pandas as pd
from datetime import date
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata, RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class ImpliedVolatility:
    def __init__(self, ticker, r, q):
        self.ticker = ticker
        self.r = r
        self.q = q
        self.c_pts = None

    def calculate_implied_volatility(self, num_expiry):
        """
        Returns the implied volatility for options
        :param num_expiry:
        :return:
        """
        def BS_call(S, K, T, r, q, sigma):
            """
            Compute from Black-Scholes the option value
            :param S: spot price
            :param K: Strike price
            :param T: Time period to maturity (yrs)
            :param r: risk-free rate
            :param q: dividend %
            :param sigma: volatility
            :return: Option price
            """
            # handle edge cases
            if T <= 0 or sigma <= 0:
                return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)

            sqrtT = np.sqrt(T)
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
            d2 = d1 - sigma * sqrtT
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        def get_IV_call(price, S, K, T, r, q):
            """
            Compute the implied volatility of a call option
            :param price: option price
            :param S: spot
            :param K: stike
            :param T: Time period
            :param r: risk-free rate
            :param q: dividend rate
            :return: implied volatility for the option
            """
            # no-arbitrage bounds for a (non-dividend) European call
            c_min = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
            c_max = S * np.exp(-q * T)

            if not (c_min < price < c_max):
                return np.nan

            return brentq(
                lambda x: BS_call(S, K, T, r, q, x) - price,
                1e-6, 5.0
            )

        # Get the expiries
        expiries = self.ticker.options
        expiries = expiries[:num_expiry]

        # Get spot rate
        spot = self.ticker.history(period="1d")["Close"].iloc[-1]

        call_pts = []
        today = date.today()
        for expiry in expiries:
            # get the calls
            chain = self.ticker.option_chain(expiry)
            calls = chain.calls.copy()

            # time to maturity
            exp_date = pd.to_datetime(expiry).date()
            T = (exp_date - today).days / 365.0
            if T <= 0:
                continue
            calls["T"] = T

            # filter out bad prices
            calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]
            # obtain the mid quoted price in bid ask
            calls["mid"] = 0.5 * (calls["bid"] + calls["ask"])

            # filter out bad moneyness
            calls["moneyness"] = calls["strike"] / spot
            calls = calls[(calls["moneyness"] > 1.0) & (calls["moneyness"] < 1.1)]

            # filter out bad OI
            calls = calls[(calls["openInterest"] > 0)]

            # compute the implied vol
            calls["IV"] = calls.apply(
                lambda row: get_IV_call(row["mid"], spot, row["strike"], T, self.r, self.q),
                axis=1
            )

            calls = calls.dropna(subset=["IV"])
            call_pts.append(calls[["moneyness", "T", "IV"]])

        self.c_pts = pd.concat(call_pts, ignore_index=True)
        return self.c_pts

    def plot_surface(self, n_moneyness, n_maturity):
        """
        Plots the implied volatility surface
        :param n_moneyness: number of moneyness points
        :param n_maturity: number of maturity points
        :return:
        """
        def interp_and_fill_binned( df, MM, TT, smooth_window=5, min_pts=3, T_bin_width=7 / 365 ):
            """
            Robust surface builder:
              1) Bin maturities into buckets (e.g., weekly)
              2) Smooth IV across moneyness within each bucket using rolling mean
              3) Interpolate on (moneyness, T) grid using linear griddata
              4) Fill holes with nearest-neighbour

            df must have columns: ["moneyness", "T", "IV"]
            MM, TT are meshgrids of moneyness and T.
            """
            df = df.copy()
            df = df.dropna(subset=["moneyness", "T", "IV"])

            # Bin maturities into buckets
            df["T_bin"] = (df["T"] / T_bin_width).round().astype(int)

            # Sort so rolling operates in moneyness order within each maturity bin
            df = df.sort_values(["T_bin", "moneyness"])

            # Smooth across moneyness inside each T-bin
            df["IV_smooth"] = (
                df.groupby("T_bin", group_keys=False)["IV"]
                .apply(lambda s: s.rolling(window=smooth_window, center=True, min_periods=min_pts).mean())
            )

            # Use smoothed values where available
            df["IV_use"] = df["IV_smooth"].fillna(df["IV"])

            # Interpolate (linear)
            IV_grid = griddata(
                points=(df["moneyness"].values, df["T"].values),
                values=df["IV_use"].values,
                xi=(MM, TT),
                method="linear"
            )

            # Fill holes with nearest neighbour
            IV_grid_nn = griddata(
                points=(df["moneyness"].values, df["T"].values),
                values=df["IV_use"].values,
                xi=(MM, TT),
                method="nearest"
            )

            return np.where(np.isnan(IV_grid), IV_grid_nn, IV_grid)

        min_moneyness, max_moneyness = self.c_pts["moneyness"].min(), self.c_pts["moneyness"].max()
        min_maturity, max_maturity = self.c_pts["T"].min() + 0.07, self.c_pts["T"].max()

        m_grid = np.linspace(min_moneyness, max_moneyness, n_moneyness)
        T_grid = np.linspace(min_maturity, max_maturity, n_maturity)
        MM, TT = np.meshgrid(m_grid, T_grid)
        # interpolate
        IV_grid = interp_and_fill_binned(self.c_pts, MM, TT)

        interp = RegularGridInterpolator(
            (T_grid, m_grid),
            IV_grid,
            bounds_error=False,
            fill_value=np.nan
        )

        def IV(m, T):
            return float(interp([[T, m]])[0])

        with plt.style.context('dark_background'):
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(TT, MM, IV_grid, cmap="viridis")
            ax.set_ylabel("Moneyness K/S")
            ax.set_xlabel("Maturity T (years)")
            # invert y and z by reversing current limits
            ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_zlim(ax.get_zlim()[::-1])
            ax.set_zlabel("Implied Vol")
            ax.set_title("Implied Volatility Surface (Calls only)")
            plt.show()
        return

