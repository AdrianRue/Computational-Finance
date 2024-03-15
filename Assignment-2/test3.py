import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



def analytical_asian_option_price_geometric(S0, K, T, r, sigma, n):
    sigma_hat = sigma * np.sqrt((2 * n + 1)  / (6 * (n + 1)))
    r_hat = ((r - 0.5 * sigma**2) + sigma_hat**2) * 0.5
    d1 = (np.log(S0 / K) + (r_hat + 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r_hat - 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
    return np.exp(-r * T) * (S0 * np.exp(r_hat * T) * norm.cdf(d1) - K * norm.cdf(d2))

def analytical_asian_option_price_arithmetic(S0, K, T, r, sigma, n):
    sigma_hat = sigma * np.sqrt((2 * n + 1)  / (6 * (n + 1)))
    r_hat = ((r - 0.5 * sigma**2) + sigma_hat**2) * 0.5
    d1 = (np.log(S0 / K) + (r_hat + 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r_hat - 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
    return np.exp(-r * T) * (S0 * np.exp((r - 0.5 * sigma**2) * T) * norm.cdf(d1) - K * norm.cdf(d2))


def MonteCarlo_asian_option_price_without_control_variate(S0, K, T, r, sigma, n, num_simulations):
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

def MonteCarlo_asian_option_price_with_control_variate(S0, K, T, r, sigma, n, num_simulations):
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
    # Calculate payoffs for both arithmetic and geometric Asian options
    payoffs_arithmetic = np.maximum(stock_prices - K, 0)
    payoffs_geometric = np.maximum(np.exp(np.mean(np.log(stock_prices))) - K, 0)
    # Calculate option prices using Monte Carlo simulation
    MC_price_arithmetic = np.exp(-r * T) * np.mean(payoffs_arithmetic)
    MC_price_geometric = np.exp(-r * T) * np.mean(payoffs_geometric)
    # Calculate control variate estimate
    control_variate_estimate = MC_price_arithmetic - (analytical_asian_option_price_arithmetic(S0, K, T, r, sigma, n) - analytical_asian_option_price_geometric(S0, K, T, r, sigma, n))
    return control_variate_estimate, MC_price_arithmetic

# parameters
S0 = 100
K = 100
T = 1
r = 0.05
sigma = 0.2
n = 1000
num_simulations = 10000

# calculate option prices
# analytical_price = analytical_asian_option_price_geometric(S0, K, T, r, sigma, n)

# MC_price = MonteCarlo_asian_option_price(S0, K, T, r, sigma, n, num_simulations)

# print('Analytical price:', analytical_price)
# print('Monte Carlo price:', MC_price)

# # Plotting
# num_simulations = np.arange(1000, 20000, 2000)
# MC_prices = [MonteCarlo_asian_option_price(S0, K, T, r, sigma, n, num) for num in num_simulations]




# plt.figure(figsize=(10, 6))
# plt.plot(num_simulations, MC_prices)
# plt.axhline(y=analytical_price, color='r', linestyle='-', label='Analytical Price')
# plt.xlabel('Number of Simulations')
# plt.ylabel('Option Price')
# plt.title('Asian Option Price as a Function of Number of Simulations')
# plt.legend()
# plt.grid(True)
# plt.show()


# Calculate option prices with control variates and collect variances
# Calculate option prices and variances both with and without control variates
# variances_without_control_variate = []
# variances_with_control_variate = []
# for num in range(1000, num_simulations + 1, 1000):
#     # Without control variate
#     MC_price_without_control_variate = MonteCarlo_asian_option_price_without_control_variate(S0, K, T, r, sigma, n, num)
#     variance_without_control_variate = np.var(MC_price_without_control_variate)
#     variances_without_control_variate.append(variance_without_control_variate)
    
#     # With control variate
#     control_variate_estimate, _ = MonteCarlo_asian_option_price_with_control_variate(S0, K, T, r, sigma, n, num)
#     variance_with_control_variate = np.var(control_variate_estimate)
#     variances_with_control_variate.append(variance_with_control_variate)

# # Plotting
# num_simulations_range = np.arange(1000, num_simulations + 1, 1000)
# plt.figure(figsize=(10, 6))
# plt.plot(num_simulations_range, variances_without_control_variate, label='Without Control Variate')
# plt.plot(num_simulations_range, variances_with_control_variate, label='With Control Variate')
# plt.xlabel('Number of Simulations')
# plt.ylabel('Variance')
# plt.title('Variance Comparison with and without Control Variates')
# plt.legend()
# plt.grid(True)
# plt.show()

# Calculate option prices with control variates and collect variances
control_variate_estimates = []
variances = []
for num in range(2000, num_simulations + 1, 1000):
    control_variate_estimate, _ = MonteCarlo_asian_option_price_with_control_variate(S0, K, T, r, sigma, n, num)
    control_variate_estimates.append(control_variate_estimate)
    variance = np.var(control_variate_estimates)
    variances.append(variance)

# Plotting
num_simulations_range = np.arange(2000, num_simulations + 1, 1000)
plt.figure(figsize=(10, 6))
plt.plot(num_simulations_range, variances, label='Variance of Control Variate Estimate')
plt.xlabel('Number of Simulations')
plt.ylabel('Variance')
plt.title('Variance Reduction with Control Variates')
plt.legend()
plt.grid(True)
plt.show()