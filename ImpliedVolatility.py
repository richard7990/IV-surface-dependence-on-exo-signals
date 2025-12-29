import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from Utilities import build_interpolated

class ImpliedVolatility:
    def __init__(self, ticker, r, q):
        self.ticker = ticker
        self.r = r
        self.q = q
        self.c_pts = None

    def _BS_call(self, S, K, T, r, q, sigma):
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

    def _get_IV_call(self, price, S, K, T, r, q):
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

        try:
            return brentq(
                lambda x: self._BS_call(S, K, T, r, q, x) - price,
                1e-6, 5.0
            )
        except Exception:
            return np.nan

    def calculate_implied_volatility(self, num_expiry, data=None):
        """
        Returns the implied volatility for options
        :param num_expiry:
        :return:
        """
        if data is not None:
            return self._calculate_from_dataframe(num_expiry)
        else:
            print("No data detected getting from yfinance")
            return self._calculate_from_yfinance(num_expiry)

    def _calculate_from_yfinance(self, num_expiry):
        my_ticker = yf.Ticker(self.ticker)
        # Get the expiries
        expiries = my_ticker.options
        if not expiries:
            return pd.DataFrame()

        expiries = expiries[:num_expiry]

        # Get spot rate
        spot = my_ticker.history(period="1d")["Close"].iloc[-1]

        call_pts = []
        today = date.today()
        for expiry in expiries:
            # get the calls
            chain = my_ticker.option_chain(expiry)
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
            # Widen the filter to capture more data
            calls = calls[(calls["moneyness"] > 1.0) & (calls["moneyness"] < 1.20)]

            # filter out bad OI
            calls = calls[(calls["openInterest"] > 0)]

            # compute the implied vol
            calls["IV"] = calls.apply(
                lambda row: self._get_IV_call(row["mid"], spot, row["strike"], T, self.r, self.q),
                axis=1
            )

            calls = calls.dropna(subset=["IV"])
            call_pts.append(calls[["moneyness", "T", "IV"]])

        rs = pd.concat(call_pts, ignore_index=True)
        self.c_pts = rs
        return rs

    def _calculate_from_dataframe(self, num_expiry):
        df = self.ticker.copy()

        # Filter for calls if column exists
        if 'call_put' in df.columns:
            df = df[df['option_type'].astype(str).str.lower().isin(['call', 'c'])]

        # Ensure datetime
        df['date'] = pd.to_datetime(df['date'])
        df['expiration'] = pd.to_datetime(df['expiration'])

        # Calculate T
        df['T'] = (df['expiration'] - df['date']).dt.days / 365.0
        df = df[df['T'] > 0]

        # Limit expiries
        if num_expiry is not None:
            unique_expiries = sorted(df['expiration'].unique())
            target_expiries = unique_expiries[:num_expiry]
            df = df[df['expiration'].isin(target_expiries)]

        # Check for underlying price
        if 'underlying_price' not in df.columns:
            if 'spot' in df.columns:
                df['underlying_price'] = df['spot']
            else:
                # If no spot price, we can't calculate moneyness properly.
                # Assuming it exists as per standard option chain data.
                raise ValueError("DataFrame must contain 'underlying_price' column")

        # Moneyness
        df['moneyness'] = df['strike'] / df['underlying_price']
        df = df[(df['moneyness'] > 1.0) & (df['moneyness'] < 1.1)]

        # Calculate IV
        if 'bid' in df.columns and 'ask' in df.columns:
            df['mid'] = 0.5 * (df['bid'] + df['ask'])
            df['IV'] = df.apply(
                lambda row: self._get_IV_call(row['mid'], row['underlying_price'], row['strike'], row['T'], self.r, self.q),
                axis=1
            )
        elif 'implied_volatility' in df.columns:
            df['IV'] = df['implied_volatility']
        else:
             raise ValueError("DataFrame must contain 'bid'/'ask' or 'implied_volatility'")

        df = df.dropna(subset=['IV'])
        self.c_pts = df[['moneyness', 'T', 'IV']]
        return self.c_pts

    def plot_surface(self, n_moneyness, n_maturity):
        """
        Plots the implied volatility surface
        :param n_moneyness: number of moneyness points
        :param n_maturity: number of maturity points
        :return:
        """
        min_moneyness, max_moneyness = self.c_pts["moneyness"].min(), self.c_pts["moneyness"].max()
        min_maturity, max_maturity = self.c_pts["T"].min() + 0.07, self.c_pts["T"].max()

        m_grid = np.linspace(min_moneyness, max_moneyness, n_moneyness)
        T_grid = np.linspace(min_maturity, max_maturity, n_maturity)
        MM, TT = np.meshgrid(m_grid, T_grid)
        # interpolate
        IV_grid = build_interpolated(self.c_pts, MM, TT)

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
