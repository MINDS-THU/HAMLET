"""
The problem involves simulating the warranty replacement process for a camera sold by Yakkon Company. The camera has a warranty period of 1.5 years, and if it fails within this period, it is replaced for free. The time until failure follows a Gamma distribution with a mean of 2.5 years and a standard deviation of 1 year. The cost to the company for each replacement is $225, and the camera is sold for $400. The objective is to estimate the number of replacements under warranty and the net present value (NPV) of profit from the sale, using a discount rate of 8%. The simulation involves generating random lifetimes for the camera and its replacements, calculating the cost to the company for replacements within the warranty period, and discounting these costs to calculate the NPV of profit.
"""

import numpy as np

def run_simulation():
    # Parameters
    iterations = 10000
    selling_price = 400.0 # selling price of camera
    replacement_cost = 225.0 # cost of replacement camera
    warranty_period = 1.5 # warranty period in years
    discount_rate = 0.08 # discount rate per year

    # Gamma distribution parameters
    mean_failure_time = 2.5
    std_failure_time = 1.0
    gamma_shape = (mean_failure_time / std_failure_time) ** 2
    gamma_scale = (std_failure_time ** 2) / mean_failure_time

    # Store results of each iteration
    num_failures_list = []
    npv_profit_list = []

    for _ in range(iterations):
        time_passed = 0.0
        num_failures = 0
        npv_profit = selling_price - replacement_cost  # initial profit from sale at time 0, undiscounted

        # Simulate the warranty process recursively
        while True:
            # Draw random failure time from Gamma distribution
            failure_time = np.random.gamma(gamma_shape, gamma_scale)
            time_passed += failure_time

            if failure_time <= warranty_period:
                num_failures += 1
                # The replacement cost occurs at the time of failure
                discount_factor = (1 + discount_rate) ** time_passed
                npv_profit -= replacement_cost / discount_factor
                # After replacement, the warranty resets, and time tracking restarts from zero.
                time_passed = 0.0
                # Keep simulating from zero time again after replacement
                continue
            else:
                # No failure within warranty, warranty ends
                break

        # Append results for current trial
        num_failures_list.append(num_failures)
        npv_profit_list.append(npv_profit)

    # Calculate performance measures
    results = {
        "failures": num_failures_list,
        "npv profit": npv_profit_list
    }

    return results


if __name__ == '__main__':
    simulation_results = run_simulation()
    print(simulation_results)