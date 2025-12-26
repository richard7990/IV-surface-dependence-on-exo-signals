
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy.stats import norm
from scipy.optimize import brentq

spy = yf.Ticker("SPY")

# Get the spot price
spot = spy.history(period="1d")["Close"].iloc[-1]
print("Spot price:", spot)

# Get the expiries
expiries = spy.options
print(expiries[:10])

def BS_call(S, K, T, r, q, sigma):
    # handle edge cases
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def get_IV(price, S, K, T, r, q):
    # no-arbitrage bounds for a (non-dividend) European call
    c_min = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    c_max = S * np.exp(-q * T)

    if not (c_min < price < c_max):
        return np.nan

    return brentq(
        lambda x: BS_call(S, K, T, r, q, x) - price,
        1e-6, 5.0
    )

r = 0.02 # risk free rate
q = 0.015  # 1.5% dividend yield (rough but fine)
today = dt.utcnow().date() # date time today
all_pts = []

for expiry in expiries:
    # get the calls and puts
    chain = spy.option_chain(expiry)
    calls = chain.calls
    puts = chain.puts

    # obtain the mid price
    calls["mid"] = 0.5 * (calls["bid"] + calls["ask"])
    puts["mid"] = 0.5 * (puts["bid"] + puts["ask"])

    # time to maturity
    exp_date = pd.to_datetime(expiry).date()
    T = (exp_date - today).days / 365.0
    if T <= 0:
        continue
    calls["T"] = T

    # filter out bad prices
    calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]
    puts = puts[(puts["bid"] > 0) & (puts["ask"] > 0)]
    # filter out bad moneyness
    calls["moneyness"] = calls["strike"]/spot
    calls = calls[(calls["moneyness"] > 0.99) & (calls["moneyness"] < 1.2)]
    # filter out bad OI
    calls = calls[(calls["openInterest"] > 0)]

    # compute the implied vol
    calls["IV"] = calls.apply(
        lambda row: get_IV(row["mid"], spot, row["strike"], T, r, q),
        axis=1
    )

    calls = calls.dropna(subset = ["IV"])
    all_pts.append(calls[["moneyness", "T", "IV"]])
pts = pd.concat(all_pts, ignore_index=True)

from scipy.interpolate import griddata

# build a grid
m_grid = np.linspace(0.95, 1.05, 31)     # moneyness axis
T_grid = np.linspace(pts["T"].min(), pts["T"].max(), 25)  # maturity axis
MM, TT = np.meshgrid(m_grid, T_grid)

# interpolate (cubic looks nice, linear is safer)
IV_grid = griddata(
    points=(pts["moneyness"].values, pts["T"].values),
    values=pts["IV"].values,
    xi=(MM, TT),
    method="linear"
)

# fill any holes with nearest-neighbour
IV_grid_nn = griddata(
    points=(pts["moneyness"].values, pts["T"].values),
    values=pts["IV"].values,
    xi=(MM, TT),
    method="nearest"
)

IV_grid = np.where(np.isnan(IV_grid), IV_grid_nn, IV_grid)


plt.figure(figsize=(8, 5))
plt.imshow(
    IV_grid,
    origin="lower",
    aspect="auto",
    extent=[m_grid.min(), m_grid.max(), T_grid.min(), T_grid.max()]
)
plt.colorbar(label="Implied Vol")
plt.xlabel("Moneyness K/S")
plt.ylabel("Maturity T (years)")
plt.title("Implied Volatility Surface (Heatmap)")
plt.show()

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(MM, TT, IV_grid)
ax.set_xlabel("Moneyness K/S")
ax.set_ylabel("Maturity T (years)")
ax.set_zlabel("Implied Vol")
plt.show()