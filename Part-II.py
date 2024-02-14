import numpy as np 
import matplotlib.pyplot as plt

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

def valueOptionMatrix(tree, T, r, K, vol):
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    columns = tree.shape[1]
    rows = tree.shape[0]

    # Print the original tree for reference
    print(tree)

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
sigma = 0.2
S = 80
T = 1.
N = 5
K = 85
r = 0.1

# Build the binomial tree
tree = buildTree(S, sigma, T, N)

# Calculate the option price using the tree
optionPrice = valueOptionMatrix(tree, T, r, K, sigma)

# Print the final option prices
print(optionPrice)
# # Play around with different ranges of N and stepsizes.

# N = np.arange(1 , 300)
# # calculate the option price for the coorect parameters
# optionPriceAnalytical = 0 # TODO 
# # calculate the option price for each n in N
# for n in N:
#     treeN = buildTree() # TODO
#     priceApproximatedly = valueOption() # TODO

# # use matplotlib to plot the analytical value
# # and the approximated value for each n