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

def average_payoff(S0, K, T, r, sigma, dt, num_simulations, num_steps):
    payoffs = np.zeros(num_simulations)

    for i in range(num_simulations):
        stock_prices = stock_sim(S0, T, r, sigma, dt, num_steps)
        payoff = np.maximum(K - stock_prices[-1], 0)
        payoffs[i] = payoff

    return np.mean(payoffs)

def run_simulation():
    S0 = 100
    K = 99
    T = 1
    r = 0.06
    sigma = 0.2
    dt = 1 / 365
    num_simulations = 100
    num_steps = int(T / dt)

    payoff = average_payoff(S0, K, T, r, sigma, dt, num_simulations, num_steps)
    option_price = np.exp(-r * T) * payoff
    print(option_price)

print(run_simulation())

def converge_payoff(S0, K, T, r, sigma, dt, num_simulations, num_steps):

    payoffs = []

    mean_payoffs = []
    i_s = []
    for i in range(num_simulations):

        stock_prices = stock_sim(S0, T, r, sigma, dt, num_steps)
        payoff = np.maximum(K - stock_prices[-1], 0)
        payoffs.append(payoff)

        if i % 10 == 0:
            mean_payoffs.append(np.mean(payoffs))
            i_s.append(i)


    return mean_payoffs, i_s

def convergence():

    S0 = 100
    K = 99
    T = 1
    r = 0.06
    sigma = 0.2
    dt = 1 / 365
    num_simulations = 5000
    num_steps = int(T / dt)

    payoffs = converge_payoff(S0, K, T, r, sigma, dt, num_simulations, num_steps)
    plt.plot(payoffs[1], payoffs[0])
    plt.show()

convergence()



