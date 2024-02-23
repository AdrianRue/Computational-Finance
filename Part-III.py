import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    call_price = S * N_d1 - K * np.exp(-r * T) * N_d2
    return call_price

def black_scholes_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

def generate_stock_market(S0, T, r, sigma_stock, dt, num_steps):
    stock_prices = np.zeros(num_steps + 1)
    stock_prices[0] = S0

    for i in range(1, num_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt))
        stock_prices[i] = stock_prices[i - 1] * (1 + r * dt + sigma_stock * np.sqrt(dt) * dW)

    return stock_prices

def euler_hedging_simulation(stock_prices, K, T, r, sigma_delta, dt, frequencies):
    num_steps = len(stock_prices) - 1
    time_points = np.linspace(0, T, num_steps + 1)
    
    plt.figure(figsize=(12, 6))

    for frequency in frequencies:
        deltas = np.zeros(num_steps + 1)
        option_values = np.zeros(num_steps + 1)
        portfolio_values = np.zeros(num_steps + 1)

        deltas[0] = black_scholes_delta(stock_prices[0], K, T, r, sigma_delta)
        option_values[0] = deltas[0] * stock_prices[0]
        portfolio_values[0] = stock_prices[0] + option_values[0]

        for i in range(1, num_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt))

            if i % frequency == 0:
                deltas[i] = black_scholes_delta(stock_prices[i], K, T - time_points[i], r, sigma_delta)
                option_values[i] = deltas[i] * stock_prices[i]
            else:
                deltas[i] = deltas[i - 1]
                option_values[i] = deltas[i] * stock_prices[i]

            portfolio_values[i] = stock_prices[i] + option_values[i]

        plt.plot(time_points, portfolio_values, label=f'Hedging Frequency: {frequency}')

    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.title('Portfolio Value Dynamics with Delta Hedging at Different Frequencies')
    plt.show()

# Parameters
S0 = 100
K = 100
T = 1
r = 0.05
sigma_stock = 0.2
sigma_delta = 0.2
dt = 1 / 252  # Daily adjustment
num_steps = 252  # Constant number of time steps
frequencies = [1, 5, 20, 252]  # Daily, Weekly, Monthly, Continuous

# Generate stock market simulation
stock_prices = generate_stock_market(S0, T, r, sigma_stock, dt, num_steps)

# Perform hedging and create plots
euler_hedging_simulation(stock_prices, K, T, r, sigma_delta, dt, frequencies)
