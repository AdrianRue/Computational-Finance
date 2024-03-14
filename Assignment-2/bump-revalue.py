import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def stock_sim(S0, T, r, sigma, dt):
    num_steps = int(T / dt)
    stock_prices = np.zeros(num_steps + 1)
    stock_prices[0] = S0
    
    for i in range(1, num_steps + 1):
        dW = np.random.normal(0, 1)
        stock_prices[i] = stock_prices[i - 1] + stock_prices[i - 1] * (r * dt + sigma * np.sqrt(dt) * dW)

    return stock_prices

def MC_option(S0, K, T, r, sigma, dt, num_simulations):
    payoffs = np.zeros(num_simulations)

    for i in range(num_simulations):
        stock_prices = stock_sim(S0, T, r, sigma, dt)
        payoff = np.maximum(stock_prices[-1] - K, 0)
        payoffs[i] = payoff

    return np.exp(-r*T)*np.mean(payoffs)

def bump_revalue(S0, K, T, r, sigma, dt, num_simulations, bump_size):
    pi0 = MC_option(S0, K, T, r, sigma, dt, num_simulations)
    S1 = S0 + bump_size
    pi1 = MC_option(S1, K, T, r, sigma, dt, num_simulations)
    return pi0, pi1

def estimate_delta(S0, K, T, r, sigma, dt, num_simulations, bump_size):
    pi0, pi1 = bump_revalue(S0, K, T, r, sigma, dt, num_simulations, bump_size)
    return (pi1 - pi0) / bump_size

# Define parameters
S0s = np.linspace(80, 120, 100)
bump_size = 0.01
K = 100
T = 1
r = 0.05
sigma = 0.2
dt = 1/252
num_simulations = 1000

# Calculate deltas for different initial stock prices
deltas = [estimate_delta(S0, K, T, r, sigma, dt, num_simulations, bump_size) for S0 in S0s]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(S0s, deltas)
plt.xlabel('Initial Stock Price')
plt.ylabel('Delta')
plt.title('Delta as a Function of Initial Stock Price')
plt.grid(True)
plt.show()


