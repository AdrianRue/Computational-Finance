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
N = 5
K = 85
r = 0.1

# # Build the binomial tree
# tree = buildTree(S, sigma, T, N)

# # Calculate the option price using the tree
# optionPrice = valueOptionMatrix(tree, T, R, K, sigma, N)

# Print the final option prices
# print(optionPrice)
# # Play around with different ranges of N and stepsizes.

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price, norm.cdf(d1)

N = np.arange(1 , 300)
# calculate the option price for the coorect parameters
optionPriceAnalytical = black_scholes_call(S, K, T, R, sigma)[0] 
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




# plt.plot(N, f0s, label='Approximated')
# plt.plot(N, optionPriceAnalytical*np.ones(len(N)), label='Analytical')
# plt.xlabel('N')
# plt.ylabel('Option Price')
# plt.legend()
# plt.show()

def hedge_param(tree, option_tree):
    delta = (option_tree[1,1] - option_tree[1,0]) / (tree[1,1] - tree[1,0])
    # print(option_tree[1,1], option_tree[1,0], tree[1,1], tree[1,0])
    return delta

sigmas = np.arange(0.01, 0.99, 0.05)
deltas = []
deltas0 = []

N=400

for sigma in sigmas:
    tree_reference  = buildTree(S, sigma, T, N)
    tree = np.copy(tree_reference)
    option_tree = valueOptionMatrix(tree, T, R, K, sigma, N)
    delta = hedge_param(tree_reference, option_tree)
    delta0 = black_scholes_call(S, K, T, R, sigma)[1]
    deltas0.append(delta0)
    deltas.append(delta)

# print(deltas)
plt.plot(sigmas, deltas, label='Approximated')
plt.plot(sigmas, deltas0, label='Analytical')
plt.xlabel('Sigma')
plt.ylabel('Delta')
plt.legend()
plt.show()

# use matplotlib to plot the analytical value
# and the approximated value for each n

def valueAmericanOptionMatrix(tree, T, r, K, vol, N):
    """Calculate the American option price using the binomial tree
    
    Arguments:
    tree -- the binomial tree
    T -- time to maturity
    r -- risk-free rate
    K -- strike price
    vol -- volatility
    N -- number of steps
    
    Returns:
    tree -- a 2D numpy array representing the binomial tree with the American option price
    """
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    columns = tree.shape[1]
    rows = tree.shape[0]
    reference = np.copy(tree)
    option_value = np.zeros((rows, columns))
    
    for c in np.arange(columns):
        S = tree[rows - 1, c]
        tree[rows - 1, c] = max(K-S, 0)
    # Walk backward, consider early exercise opportunities
    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = tree[i + 1, j]
            up = tree[i + 1, j + 1]
            immediate_exercise_payoff = max(K - reference[i, j], 0)
           
            option_value = np.exp(-r * dt) * (p * up + (1 - p) * down)
            #print("Immediate ex: {}, opt val: {}, stock {}, Down: {}, UP: {}" .format(immediate_exercise_payoff, option_value, tree[i, j], down, up))
            tree[i, j] = max(immediate_exercise_payoff, option_value)
    return tree
    
