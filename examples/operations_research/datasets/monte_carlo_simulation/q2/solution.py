import numpy as np

def run_simulation():
    # Simulation parameters
    num_iterations = 10000
    num_days = 10
    order_quantities = range(6, 13)

    # Demand probabilities and outcomes per day
    daily_demands = [0, 1, 2]
    daily_probs = [0.5, 0.3, 0.2]

    # Profit parameters
    sale_profit = 600 - 350  # Profit per unit sold = 250
    salvage_loss = 350 - 250 # Loss per unsold unit = 100

    # Store results
    results = {}
    res = {}
    for Q in order_quantities:
        # Simulate all iterations at once for efficiency:
        # Generate a (num_iterations x num_days) array of daily demands
        sim_demands = np.random.choice(daily_demands, size=(num_iterations, num_days), p=daily_probs)
        
        # Total demand per simulation run
        total_demand = sim_demands.sum(axis=1)
        
        # Number of units sold is the minimum of total demand and the order quantity Q
        units_sold = np.minimum(Q, total_demand)
        
        # Calculate profit for each simulation run:
        # Profit = 250 * (units sold) - 100 * (unsold units)
        profit = sale_profit * units_sold - salvage_loss * (Q - units_sold)
        
        # Compute statistics: mean and standard deviation of profit
        mean_profit = profit.mean()
        std_profit = profit.std()
        
        results[Q] = (mean_profit, std_profit)
        res[f"profit for order quantity {Q}"] = profit
        print(f"Order Quantity {Q}: Mean Profit = ${mean_profit:.2f}, Std Dev = ${std_profit:.2f}")

    # Optionally, print the results dictionary
    print("\nResults (Order Quantity: (Mean, Std Dev))")
    print(results)
    return res

if __name__ == "__main__":
    res = run_simulation()
    print(res)
