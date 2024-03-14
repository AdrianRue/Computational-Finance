import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def analytical_asian_option_price(S0, K, T, r, sigma, n):
    sigma_hat = sigma * np.sqrt((2 * n + 1)  / (6 * (n + 1)))
    r_hat = ((r - 0.5 * sigma**2) + sigma_hat**2) * 0.5
    d1 = (np.log(S0 / K) + (r_hat + 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r_hat - 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
    return np.exp(-r * T) * (S0 * np.exp(r_hat * T) * norm.cdf(d1) - K * norm.cdf(d2))

def MonteCarlo_asian_option_price(S0, K, T, r, sigma, n, num_simulations):
    dt = T / n
    stock_prices = np.zeros(num_simulations)
    for i in range(num_simulations):
        stock_price = S0
        A = S0
        for j in range(n):
            dW = np.random.normal(0, np.sqrt(dt))
            stock_price = stock_price * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW)
            A += stock_price
        A /= n + 1
        stock_prices[i] = A
    payoffs = np.maximum(stock_prices - K, 0)
    return np.exp(-r * T) * np.mean(payoffs)

# parameters
S0 = 100
K = 100
T = 1
r = 0.05
sigma = 0.2
n = 1000
num_simulations = 100000

# calculate option prices
analytical_price = analytical_asian_option_price(S0, K, T, r, sigma, n)

MC_price = MonteCarlo_asian_option_price(S0, K, T, r, sigma, n, num_simulations)

print('Analytical price:', analytical_price)
print('Monte Carlo price:', MC_price)

# Plotting
num_simulations = np.arange(1000, 10000, 2000)
MC_prices = [MonteCarlo_asian_option_price(S0, K, T, r, sigma, n, num) for num in num_simulations]




plt.figure(figsize=(10, 6))
plt.plot(num_simulations, MC_prices)
plt.axhline(y=analytical_price, color='r', linestyle='-', label='Analytical Price')
plt.xlabel('Number of Simulations')
plt.ylabel('Option Price')
plt.title('Asian Option Price as a Function of Number of Simulations')
plt.legend()
plt.grid(True)
plt.show()

