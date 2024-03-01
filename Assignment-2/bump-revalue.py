import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def stock_sim(S0, T, r, sigma, dt, num_steps):
    stock_prices = np.zeros(num_steps + 1)
    stock_prices[0] = S0
    num_steps = int(T / dt)
    
    for i in range(1, num_steps + 1):
        dW = np.random.normal(0, 1)
        stock_prices[i] = stock_prices[i - 1] + stock_prices[i - 1] * (r * dt + sigma * np.sqrt(dt) * dW)

    return stock_prices

def MC_option(S0, K, T, r, sigma, dt, num_simulations, num_steps):
    payoffs = np.zeros(num_simulations)

    for i in range(num_simulations):
        stock_prices = stock_sim(S0, T, r, sigma, dt, num_steps)
        payoff = np.maximum(stock_prices[-1] - K, 0)
        payoffs[i] = payoff

    return np.mean(payoffs)

def bump_revalue(p_vec, bump_size, bump):
    pi0 = MC_option(p_vec[0], p_vec[1], p_vec[2], p_vec[3], p_vec[4], p_vec[5], p_vec[6], p_vec[7])
    p_vec[bump] += bump_size
    pi1 = MC_option(p_vec[0], p_vec[1], p_vec[2], p_vec[3], p_vec[4], p_vec[5], p_vec[6], p_vec[7])
    
