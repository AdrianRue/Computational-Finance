import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm
from timer import Timer

def buildTree(S, vol, T, N):
    dt = T / N
    matrix = np.zeros((N + 1, N + 1))
    matrix[0, 0] = S
    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))
    
    for i in np.arange(N + 1):
        for j in np.arange(i + 1):
            matrix[i, j] = S * u**(j) * d**(i - j)
    return matrix

def valueOptionMatrix(tree, T, r, K, vol, N):
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    columns = tree.shape[1]
    rows = tree.shape[0]

    # Print the original tree for reference
    #print(tree)

    # Walk backward, add the payoff function in the last row
    for c in np.arange(columns):
        S = tree[rows - 1, c]
        tree[rows - 1, c] = max(S - K, 0)

    # For all other rows, combine from previous rows
    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = tree[i + 1, j]
            up = tree[i + 1, j + 1]
            tree[i, j] = np.exp(-r * dt) * (p * up + (1 - p) * down)

    return tree

# Parameters
sigma = 0.5
S = 100
T = 1.
N = 50
K = 99
r = 0.06

# Build the binomial tree
tree = buildTree(S, sigma, T, N)

# Calculate the option price using the tree
optionPrice = valueOptionMatrix(tree, T, r, K, sigma, N)

# Print the final option prices
print(optionPrice)
# # Play around with different ranges of N and stepsizes.

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

N = np.arange(1 , 300)
# calculate the option price for the coorect parameters
optionPriceAnalytical = black_scholes_call(S, K, T, r, sigma) 
# calculate the option price for each n in N
f0s = []
time_run = []
for n in N:
    
    t = Timer()
    t.start()
    treeN = buildTree(S, sigma, T, n) # TODO
    priceApproximatedly = valueOptionMatrix(treeN, T, r, K, sigma, n)
    time_run.append(t.stop())
    f0s.append(priceApproximatedly[0,0])



plt.plot(N, f0s, label='Approximated')
plt.plot(N, optionPriceAnalytical*np.ones(len(N)), label='Analytical')
plt.xlabel('N')
plt.ylabel('Option Price')
plt.legend()
plt.show()

# use matplotlib to plot the analytical value
# and the approximated value for each n

plt.plot(N, time_run, '.')
plt.plot(N, np.poly1d(np.polyfit(N, time_run, 2))(N))
plt.xlabel('N')
plt.ylabel('time of simulation')
plt.show()

S = 100
T = 1.
N = np.arange(1 , 300)
K = 99
r = 0.06
sigma = np.arange(0.05, 0.5, 0.05)



# Checking how many time steps it takes certain sigmas to converge
for vol in sigma:


    res_prev = 0

    for n in N:

        treeN = buildTree(S, vol, T, n)
        priceApproximatedly = valueOptionMatrix(treeN, T, r, K, vol, n)
        diff = (abs(priceApproximatedly[0,0] - res_prev)) / ((priceApproximatedly[0,0] + res_prev) / 2) * 100
        res_prev = priceApproximatedly[0,0]

        # If change in percentage is within 0.01%
        if diff < 0.1:
            print("Volatility", vol, "converges after", n, "time steps")
            break

