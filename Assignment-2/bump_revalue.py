import numpy as np
import matplotlib.pyplot as plt


def call_option_simulation_1(S0, K, T, r, sigma, iterations):
    np.random.seed(1)
    option_data = np.zeros([iterations, 2])

    rand = np.random.normal(0, 1, [1, iterations])

    stock_price = S0*np.exp(T*(r - 0.5*sigma**2) + sigma*np.sqrt(T)*rand)

    option_data[:, 1] = stock_price - K

    average = np.sum(np.amax(option_data, axis=1))/float(iterations)

    return np.exp(-1.0*r*T)*average

def call_option_simulation_2(S0, K, T, r, sigma, iterations):
    np.random.seed(100)
    option_data = np.zeros([iterations, 2])

    rand = np.random.normal(0, 1, [1, iterations])

    stock_price = S0*np.exp(T*(r - 0.5*sigma**2) + sigma*np.sqrt(T)*rand)

    option_data[:, 1] = stock_price - K

    average = np.sum(np.amax(option_data, axis=1))/float(iterations)

    return np.exp(-1.0*r*T)*average

def bump_and_revalue_same(S0, K, T, r, sigma, iterations, bump_size):
    pi_0_same = call_option_simulation_1(S0, K, T, r, sigma, iterations)
    pi_1_same = call_option_simulation_1(S0 + bump_size, K, T, r, sigma, iterations)

    return pi_0_same, pi_1_same

def bump_and_revalue_diff(S0, K, T, r, sigma, iterations, bump_size):
    pi_0_diff = call_option_simulation_1(S0, K, T, r, sigma, iterations)
    pi_1_diff = call_option_simulation_2(S0 + bump_size, K, T, r, sigma, iterations)

    return pi_0_diff, pi_1_diff

def estimate_delta_same(S0, K, T, r, sigma, iterations, bump_size):
    pi_0, pi_1 = bump_and_revalue_same(S0, K, T, r, sigma, iterations, bump_size)
    return (pi_1 - pi_0)/bump_size

def estimate_delta_diff(S0, K, T, r, sigma, iterations, bump_size):
    pi_0, pi_1 = bump_and_revalue_diff(S0, K, T, r, sigma, iterations, bump_size)
    return (pi_1 - pi_0)/bump_size

# Define parameters
S0s = np.linspace(80, 120, 100)
bump_size = 0.1
K = 99
T = 1
r = 0.05
sigma = 0.2
iterations = 50000

# Calculate deltas for different initial stock prices
deltas_same = [estimate_delta_same(S0, K, T, r, sigma, iterations, bump_size) for S0 in S0s]
deltas_diff = [estimate_delta_diff(S0, K, T, r, sigma, iterations, bump_size) for S0 in S0s]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(S0s, deltas_same, label = 'Same Seed')
plt.plot(S0s, deltas_diff, label = 'Different Seed')
plt.xlabel('Initial Stock Price')
plt.ylabel('Delta')
plt.title('Delta as a Function of Initial Stock Price')
plt.legend()
plt.grid(True)
plt.show()
