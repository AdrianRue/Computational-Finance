import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def call_option_simulation(S0, K, T, r, sigma, iterations):
    option_data = np.zeros([iterations, 2])

    rand = np.random.normal(0, 1, [1, iterations])

    stock_price = S0*np.exp(T*(r - 0.5*sigma**2) + sigma*np.sqrt(T)*rand)

    option_data[:, 1] = stock_price - K

    average = np.sum(np.amax(option_data, axis=1))/float(iterations)

    return np.exp(-1.0*r*T)*average

def bump_and_revalue(S0, K, T, r, sigma, iterations, bump_size):
    pi_0 = call_option_simulation(S0, K, T, r, sigma, iterations)
    pi_1 = call_option_simulation(S0 + bump_size, K, T, r, sigma, iterations)

    return pi_0, pi_1

def estimate_delta(S0, K, T, r, sigma, iterations, bump_size):
    pi_0, pi_1 = bump_and_revalue(S0, K, T, r, sigma, iterations, bump_size)
    return (pi_1 - pi_0)/bump_size

# Define parameters
S0s = np.linspace(80, 120, 100)
bump_size = 1
K = 100
T = 1
r = 0.05
sigma = 0.2
iterations = 1000

# Calculate deltas for different initial stock prices
deltas = [estimate_delta(S0, K, T, r, sigma, iterations, bump_size) for S0 in S0s]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(S0s, deltas)
plt.xlabel('Initial Stock Price')
plt.ylabel('Delta')
plt.title('Delta as a Function of Initial Stock Price')
plt.grid(True)
plt.show()
