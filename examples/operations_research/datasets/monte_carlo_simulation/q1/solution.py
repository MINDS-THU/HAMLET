import numpy as np

def run_simulation():
    # Parameters
    S0 = 100          # initial stock price
    r = 0.09          # risk-free rate
    sigma = 0.30      # volatility
    T = 1.0           # time to maturity in years
    n_steps = 52      # weekly steps over one year
    dt = T / n_steps  # time step size
    strike = 110      # exercise price
    n_simulations = 10000  # number of simulation runs

    # Generate random draws for each time step and each simulation
    rand = np.random.randn(n_simulations, n_steps)

    # Calculate the drift and diffusion components for each time step
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * rand

    # Simulate price paths (each row is one simulation)
    increments = np.exp(drift + diffusion)
    prices = S0 * np.cumprod(increments, axis=1)

    # Compute the arithmetic average of prices for each simulation
    avg_prices = np.mean(prices, axis=1)

    # Calculate the option payoff for each simulation: max(avg - strike, 0)
    payoffs = np.maximum(avg_prices - strike, 0)

    # Discount the payoffs back to present value using the risk-free rate
    discounted_payoffs = np.exp(-r * T) * payoffs

    # Compute the mean and standard deviation of the discounted payoffs
    mean_value = np.mean(discounted_payoffs)
    std_value = np.std(discounted_payoffs)

    print("Mean of Asian Option Value:", mean_value)
    print("Standard Deviation of Asian Option Value:", std_value)

    return {"asian option value": discounted_payoffs}

if __name__ == "__main__":
    res = run_simulation()
    print(res)