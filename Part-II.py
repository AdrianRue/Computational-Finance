import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm

def buildTree(S, vol, T, N):
    """Build a binomial tree for the stock price
    
    Arguments:
    S -- initial stock price
    vol -- volatility
    T -- time to maturity
    N -- number of steps
    
    Returns:
    matrix -- a 2D numpy array representing the binomial tree
    
    """
    dt = T / N
    matrix = np.zeros((N + 1, N + 1))
    matrix[0, 0] = S
    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))
    
    # Fill the matrix
    for i in np.arange(N + 1):
        for j in np.arange(i + 1):
            matrix[i, j] = S * u**(j) * d**(i - j)
    return matrix

def valueOptionMatrix(tree, T, r, K, vol, N):
    """Calculate the option price using the binomial tree
    
    Arguments:
    tree -- the binomial tree
    T -- time to maturity
    r -- risk-free rate
    K -- strike price
    vol -- volatility
    N -- number of steps
    
    Returns:
    tree -- a 2D numpy array representing the binomial tree with the option price
    
    """
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    columns = tree.shape[1]
    rows = tree.shape[0]

    # Print the original tree for reference
    # print(tree)

    # Walk backward, add the payoff function in the last row
    for c in np.arange(columns):
        S = tree[rows - 1, c]
        tree[rows - 1, c] = max(K-S, 0)

    # For all other rows, combine from previous rows
    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = tree[i + 1, j]
            up = tree[i + 1, j + 1]
            tree[i, j] = np.exp(-r * dt) * (p * up + (1 - p) * down)


    return tree

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
            print("Immediate ex: {}, opt val: {}, stock {}, Down: {}, UP: {}" .format(immediate_exercise_payoff, option_value, tree[i, j], down, up))
            tree[i, j] = max(immediate_exercise_payoff, option_value)

    return tree
    

# Parameters
sigma = 0.2
S = 100
T = 1.
N = 5
K = 99
R = 0.06

# # Build the binomial tree
# tree = buildTree(S, sigma, T, N)

# # Calculate the option price using the tree
# optionPrice = valueOptionMatrix(tree, T, R, K, sigma, N)

# Print the final option prices
# print(optionPrice)
# # Play around with different ranges of N and stepsizes.

def black_scholes_call(S, K, T, r, sigma):
    """Calculate the price of a European call option using the Black-Scholes formula
    
    Arguments:
    S -- initial stock price
    K -- strike price
    T -- time to maturity
    r -- risk-free rate
    sigma -- volatility
    
    Returns:
    call_price -- the price of the call option
    delta_0 -- the analytical hedge parameter

    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta_0 = norm.cdf(d1)
    call_price = S * delta_0 - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price, delta_0

N = np.arange(1 , 300)
# calculate the option price for the coorect parameters
optionPriceAnalytical = black_scholes_call(S, K, T, R, sigma)[0] 
# calculate the option price for each n in N
# f0s = []
# for n in N:
#     treeN = buildTree(S, sigma, T, n) 
#     priceApproximatedly = valueOptionMatrix(treeN, T,R,K,sigma, n)
#     f0s.append(priceApproximatedly[0,0])


# plt.plot(N, f0s, label='Approximated')
# plt.plot(N, optionPriceAnalytical*np.ones(len(N)), label='Analytical')
# plt.xlabel('N')
# plt.ylabel('Option Price')
# plt.legend()
# plt.show()

def hedge_param(tree, option_tree):
    """Calculate the hedge parameter delta using the binomial tree
    
    Arguments:
    tree -- the binomial tree
    option_tree -- the binomial tree with the option price
    
    Returns:
    delta -- the hedge parameter

    """
    delta = (option_tree[1,1] - option_tree[1,0]) / (tree[1,1] - tree[1,0])
    return delta

sigmas = np.arange(0.02, 0.99, 0.05)
deltas = []
deltas0 = []

N=90

# using European options
# for sigma in sigmas:
#     tree_reference  = buildTree(S, sigma, T, N)
#     tree = np.copy(tree_reference)
#     option_tree = valueOptionMatrix(tree, T, R, K, sigma, N)
#     delta = hedge_param(tree_reference, option_tree)
#     delta0 = black_scholes_call(S, K, T, R, sigma)[1]
#     deltas0.append(delta0)
#     deltas.append(delta)

# using American options
# for sigma in sigmas:
#     tree_reference  = buildTree(S, sigma, T, N)
#     tree = np.copy(tree_reference)
#     option_tree = valueAmericanOptionMatrix(tree, T, R, K, sigma, N)
#     delta = hedge_param(tree_reference, option_tree)
#     delta0 = black_scholes_call(S, K, T, R, sigma)[1]
#     deltas0.append(delta0)
#     deltas.append(delta)



# print(deltas)
# plt.plot(sigmas, deltas, label='Approximated')
# plt.plot(sigmas, deltas0, label='Analytical')
# plt.xlabel('Sigma')
# plt.ylabel('Delta')
# plt.legend()
# plt.show()

N = 5
sigma =.2
print(buildTree(S, sigma, T, N))
print('EUROP', valueOptionMatrix(buildTree(S, sigma, T, N), T, R, K, sigma, N))
print('AMERICAN', valueAmericanOptionMatrix(buildTree(S, sigma, T, N), T, R, K, sigma, N))

# tree = np.array([1,2,3]) # tree = [1,2,3]
# option_tree = valueOptionMatrix(tree) # tree = [1,4,3] option_tree = [1,4,3]
# valueOptionMatrix(tree):
# tree[1] = 4
# return tree
# print("MANUALTREE", buildTree(S, 0.2, T, 5))
# print("MANUALOPTTREE", valueOptionMatrix(buildTree(S, 0.2, T, 5), T, R, K, 0.2, 5))




# use matplotlib to plot the analytical value
# and the approximated value for each n