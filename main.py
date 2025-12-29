import yfinance as yf
from ImpliedVolatility import ImpliedVolatility
from ReadOptionData import get_option_data

spy = "SPY"
#new_spy = get_option_data("SPY")
r = 0.02 # risk free rate
q = 0.015  # 1.5% dividend yield (rough but fine)
num_expiries = 100

iv = ImpliedVolatility(spy, r, q)
tt = iv.calculate_implied_volatility(num_expiries)
iv.plot_surface(n_moneyness=30, n_maturity=30)

n_moneyness=30
n_maturity=30
c_pts = tt