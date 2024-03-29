import numpy as np 
import matplotlib.pyplot as plt

def initialize_grid(N, M, T, S0, Smax, K, r, sigma):
    dtau = T / M
    ds = Smax / N

    # Initialize the grid
    grid = np.zeros((N + 1, M + 1))
    grid.fill(0)
    
    # Set up stock price axis starting from S0
    S = np.linspace(S0, Smax, N + 1)
    
    # Initialize option values based on boundary conditions
    grid[:, 0] = np.maximum(S - K, 0)  # Intrinsic value at t=0
    
    # Set boundary conditions
    grid[0, :] = 0  # Lower boundary
    grid[-1, :] = Smax - K * np.exp(-r * np.linspace(0, T, M + 1))  # Upper boundary

    return grid, dtau, ds, S

def FTCS(grid, N, M, dtau, ds, r, sigma):
    gamma = (r - 0.5 * sigma ** 2) * dtau / (2 * ds)
    alpha = 0.5 * sigma ** 2 * dtau / (ds ** 2)
    beta = r * dtau

    # Loop through the grid
    for j in range(1, M + 1):
        for i in range(1, N):
            grid[i, j] = (gamma + alpha) * grid[i + 1, j - 1] + (1 - 2 * alpha - beta) * grid[i, j - 1] + (alpha - gamma) * grid[i - 1, j - 1]

    return grid

def plot_grid(grid, N, M, S, T):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(S, np.linspace(0, T, M + 1))
    ax.plot_surface(X, Y, grid.T, cmap='viridis')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Time to Expiry')
    ax.set_zlabel('Option Price')
    plt.show()

def main():
    # Parameters
    N = 100
    M = 100
    T = 1
    S0_in_money = 100
    S0_at_money = 110
    S0_out_money = 120
    Smax = 200
    K = 110
    r = 0.04
    sigma = 0.3

    # Initialize the grid for in-the-money option
    grid_in_money, dtau, ds, S = initialize_grid(N, M, T, S0_in_money, Smax, K, r, sigma)
    # Run the FTCS scheme for in-the-money option
    grid_in_money = FTCS(grid_in_money, N, M, dtau, ds, r, sigma)
    # Plot the grid for in-the-money option
    plot_grid(grid_in_money, N, M, S, T)

    # Repeat the above process for at-the-money and out-of-the-money options
    # Initialize the grid for at-the-money option
    grid_at_money, _, _, _ = initialize_grid(N, M, T, S0_at_money, Smax, K, r, sigma)
    grid_at_money = FTCS(grid_at_money, N, M, dtau, ds, r, sigma)
    plot_grid(grid_at_money, N, M, S, T)

    # Initialize the grid for out-of-the-money option
    grid_out_money, _, _, _ = initialize_grid(N, M, T, S0_out_money, Smax, K, r, sigma)
    grid_out_money = FTCS(grid_out_money, N, M, dtau, ds, r, sigma)
    plot_grid(grid_out_money, N, M, S, T)

if __name__ == '__main__':
    main()
