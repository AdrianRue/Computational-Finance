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

def euler_method_simulation(S0, K, T, r, sigma, frequency):
    dt = T / frequency
    steps = int(T * frequency)
    S = np.zeros(steps + 1)
    delta = np.zeros(steps + 1)
    option_value = np.zeros(steps + 1)
    PnL = np.zeros(steps + 1)

    S[0] = S0
    option_value[0] = black_scholes_call(S0, K, T, r, sigma)

    for i in range(1, steps + 1):
        delta[i - 1] = black_scholes_call_delta(S[i - 1], K, T - i * dt, r, sigma)
        S[i] = S[i - 1] + r * S[i - 1] * dt + sigma * S[i - 1] * np.sqrt(dt) * np.random.normal()
        option_value[i] = black_scholes_call(S[i], K, T - i * dt, r, sigma)
        PnL[i] = delta[i - 1] * (S[i] - S[i - 1])

    return S, delta, option_value, PnL

def black_scholes_call_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

# Example usage:
S0 = 100
K = 99
T = 1
r = 0.06
sigma = 0.2

# Plotting
def plot_results(S, delta, PnL, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(S, label='Stock Price')
    plt.title(title)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(delta, label='Delta')
    plt.plot(PnL, label='PnL')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.title('Delta and PnL over Time')
    plt.legend()

    plt.show()

# Volatility Matching Experiment
frequency_list = [1, 5, 7]  # Daily, Weekly, Monthly adjustments

for freq in frequency_list:
    S, delta, option_value, PnL = euler_method_simulation(S0, K, T, r, sigma, freq)
    plot_results(S, delta, PnL, f'Hedging Frequency: {freq} adjustments per week')

# Volatility Mismatch Experiment
sigma_mismatch_list = [0.1, 0.3, 0.5]  # Mismatch levels

for sigma_mismatch in sigma_mismatch_list:
    S, delta, option_value, PnL = euler_method_simulation(S0, K, T, r, sigma + sigma_mismatch, 1)  # Daily adjustments
    plot_results(S, delta, PnL, f'Volatility Mismatch: {sigma_mismatch}')
