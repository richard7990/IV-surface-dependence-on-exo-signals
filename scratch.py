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

