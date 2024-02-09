import numpy as np 
import matplotlib.pyplot as plt

def buildTree(S, vol, T, N):
    dt = T / N
    matrix = np.zeros((N + 1 , N + 1))
    matrix[0,0] = S
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))
    # Iterate over the lower triangle

    for i in np.arange(N + 1 ): # iiterate over rows
        for j in np.arange( i + 1 ): # iterate overs columns
    # Hint : express each cell as a combination of up
    # and down moves
            matrix[i ,j] = S*u**(j)*d**(i-j)
            return matrix
        
sigma = 0.1
S = 80
T = 1. 
N = 2
Tree = buildTree(S, sigma , T, N)     
print(Tree)  

# def valueOptionMatrix(tree , T, r , K, vol ):
#     dt = T / N
#     u = 0 # TODO
#     d = 0 # TODO
#     p = 0 # TODO
#     columns = tree.shape[ 1 ]
#     rows = tree.shape[ 0 ]
#     # Walk backward , we start in last row of the matrix
#     # Add the payoff function in the last row
#     for c in np.arange( columns ):
#         S = tree[rows - 1 , c ] # value in the matrix
#         tree[rows - 1 , c ] = 0 # TODO

#     # For all other rows , we need to combine from previous rows
#     # We walk backwards , from the last row to the first row
#     for i in np.arange(rows - 1 )[ : : - 1 ]:
#         for j in np.arange( i + 1 ):
#         down = tree[i + 1 , j ]
#         up = tree[i + 1 , j + 1 ]
#         tree[i , j ] = 0 # TODO
#     return tree

# sigma = 0.1
# S = 80
# T = 1.
# N = 2
# K = 85
# r = 0.1
# tree = buildTree(S , sigma , T, N)
# valueOptionMatrix(tree , T, r , K, sigma)

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