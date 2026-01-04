import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
            return self._calculate_from_dataframe(num_expiry, data)
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

    def _calculate_from_dataframe(self, num_expiry, data):
        def get_spot_series(ticker: str, start_date: str, end_date: str) -> pd.Series:
            px = yf.download(
                ticker,
                start=start_date,
                end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                progress=False
            )
            if px.empty:
                raise ValueError(f"No underlying price data returned for {ticker} between {start_date} and {end_date}")
            # convert index to datetime
            px.index = pd.to_datetime(px.index)
            px = pd.Series(px['Close'].values.ravel(), name="spot", index=px.index).dropna()
            return px

        df = data.copy()

        # Filter for calls if column exists
        if 'call_put' in df.columns:
            df = df[df['call_put'].astype(str).str.lower().isin(['call', 'c'])]

        # Ensure datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['expiration'] = pd.to_datetime(df['expiration'], format='%Y-%m-%d')

        # Calculate T
        df['T'] = (df['expiration'] - df['date']).dt.days / 365.0
        df = df[df['T'] > 10/365] # ultra short maturities
        df = df.groupby("T").filter(lambda x: len(x) >= 5)

        # Limit expiries
        if num_expiry is not None:
            unique_expiries = sorted(df['expiration'].unique())
            target_expiries = unique_expiries[:num_expiry]
            df = df[df['expiration'].isin(target_expiries)]

        # Check for underlying price
        if 'spot' not in df.columns:
            # get the spot data from yfinance
            spot_series = get_spot_series(self.ticker, df['date'].min().strftime("%Y-%m-%d"), df['date'].max().strftime("%Y-%m-%d"))
        else:
            raise KeyError('spot must be in the coloumns of dataframe')

        # Moneyness
        # Ensure everything is datetime and sorted
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        spot_df = spot_series.rename("spot").to_frame().reset_index()
        spot_df.columns = ["date", "spot"]
        spot_df["date"] = pd.to_datetime(spot_df["date"]).dt.normalize()
        spot_df = spot_df.sort_values("date")

        df = df.sort_values("date")

        # As-of join: closest spot at or before the option date
        df = pd.merge_asof(
            df,
            spot_df,
            on="date",
            direction="backward",  # <= this is the key
            tolerance=pd.Timedelta("7D")  # optional safety
        )

        df = df.dropna(subset=['spot'])

        df['moneyness'] = df['strike'] / df['spot']
        df = df[(df['moneyness'] > 1.0) & (df['moneyness'] < 1.1)]

        # Calculate IV
        if 'bid' in df.columns and 'ask' in df.columns:
            df['mid'] = 0.5 * (df['bid'] + df['ask'])
            df['IV'] = df.apply(
                lambda row: self._get_IV_call(row['mid'], row['spot'], row['strike'], row['T'], self.r, self.q),
                axis=1
            )
        elif 'implied_volatility' in df.columns:
            df['IV'] = df['implied_volatility']
        else:
             raise ValueError("DataFrame must contain 'bid'/'ask' or 'implied_volatility'")

        df = df.dropna(subset=['IV'])
        self.c_pts = df[['date', 'expiration','moneyness', 'T', 'IV']]
        return self.c_pts

    def plot_surface(self, IV_grid, T_mesh, M_mesh):
        """
        Plots the implied volatility surface
        :param n_moneyness: number of moneyness points
        :param n_maturity: number of maturity points
        :return:
        """

        with plt.style.context('dark_background'):
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(T_mesh, M_mesh, IV_grid, cmap="viridis")
            ax.set_ylabel("Moneyness K/S")
            ax.set_xlabel("Maturity T (years)")
            # invert y and z by reversing current limits
            ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_zlim(ax.get_zlim()[::-1])
            ax.set_zlabel("Implied Vol")
            ax.set_title("Implied Volatility Surface (Calls only)")
            plt.show()
        return

    def plot_heatmap(self, IV_grid, T_mesh, M_mesh):
        """
        Plots the implied volatility surface as a heatmap
        """
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots(figsize=(10, 6))
            # Use contourf for a smooth heatmap representation
            cp = ax.contourf(T_mesh, M_mesh, IV_grid, levels=100, cmap='viridis')
            cbar = fig.colorbar(cp)
            cbar.set_label('Implied Volatility')

            ax.set_xlabel("Maturity T (years)")
            ax.set_ylabel("Moneyness K/S")
            ax.set_title("Implied Volatility Heatmap")
            plt.show()
