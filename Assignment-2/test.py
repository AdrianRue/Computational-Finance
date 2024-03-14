import numpy as np

def stock_sim(S0, T, r, sigma, dt, num_steps):
    stock_prices = np.zeros(num_steps + 1)
    stock_prices[0] = S0
    num_steps = int(T / dt)
    
    for i in range(1, num_steps):
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


# Function to calculate option price using Monte Carlo simulation
def calculate_option_price(seed=None):
    # Replace this function with your own implementation
    # Simulate option price using Monte Carlo method
    if seed:
        np.random.seed(seed)
    # Your Monte Carlo simulation code here
    option_price = MC_option(100, 100, 1, 0.05, 0.2, 1/252, 1000, 100)
    return option_price

# Function to estimate hedge parameter δ using bump-and-revalue method
def estimate_delta(p, bump_size, parameter_to_bump, approach='different_seeds'):
    # Calculate original option price
    original_price = calculate_option_price()
    
    if approach == 'different_seeds':
        # Generate different random seeds for bumped and unbumped estimates
        bumped_seed = np.random.randint(1, 1000)
        unbumped_seed = np.random.randint(1001, 2000)
        
        # Calculate bumped and unbumped option prices
        bumped_price = calculate_option_price(bumped_seed)
        unbumped_price = calculate_option_price(unbumped_seed)
    else:
        # Generate a single random seed for both estimates
        seed = np.random.randint(1, 1000)
        
        # Calculate unbumped option price using the generated seed
        unbumped_price = calculate_option_price(seed)
        
        # Recalculate option price with the bumped parameter using the same seed
        bumped_price = calculate_option_price(seed)
    
    # Estimate delta using finite difference formula
    delta = (bumped_price - unbumped_price) / bump_size
    
    return delta

# Test the functions
p = 100  # Initial parameter value
bump_size = 0.01  # Bump size
parameter_to_bump = 'S_0'  # Parameter to bump (e.g., underlying asset price)

# Approach 1: Using different seeds
delta_approach1 = estimate_delta(p, bump_size, parameter_to_bump, approach='different_seeds')
print("Delta using different seeds:", delta_approach1)

# Approach 2: Using the same seed
delta_approach2 = estimate_delta(p, bump_size, parameter_to_bump, approach='same_seed')
print("Delta using the same seed:", delta_approach2)
