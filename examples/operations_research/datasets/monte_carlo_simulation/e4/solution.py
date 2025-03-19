"""
The problem involves simulating the cash flows from a new car model over a five-year period to estimate the NPV of after-tax cash flows. The simulation uses a Monte Carlo approach to account for uncertainties in sales and annual decay rates, which are modeled using triangular distributions. The fixed development cost, margin per car, tax rate, and discount rate are given as constants. The simulation involves calculating unit sales, unit contribution, revenue minus variable cost, depreciation, before-tax profit, after-tax profit, and cash flow for each year. The NPV is then calculated using the discount rate and the cash flows over the five years.
"""

import numpy as np

def run_simulation():
    # Constants
    fixed_cost = 700_000_000
    initial_margin = 4000
    decrease_margin_rate = 0.04
    tax_rate = 0.4
    discount_rate = 0.1

    years = 5

    # Triangular distribution parameters
    sales_params = (50_000, 75_000, 85_000)
    decay_rate_params = (0.05, 0.08, 0.1)

    # Simulation function
    def simulate_npv(num_simulations=10000):
        npvs = []
        cash_flow_ls = []
        for _ in range(num_simulations):
            # Simulate sales for year 1
            sales = np.random.triangular(*sales_params)
            
            # Calculate unit contributions for each year
            contributions = [initial_margin * ((1 - decrease_margin_rate) ** i) for i in range(years)]
            
            # Calculate sales and cash flows for each year
            cash_flows = []
            for year in range(years):
                if year > 0:
                    decay_rate = np.random.triangular(*decay_rate_params)
                    sales *= (1 - decay_rate)
                
                revenue_minus_cost = sales * contributions[year]
                depreciation = fixed_cost / years
                before_tax_profit = revenue_minus_cost - depreciation
                after_tax_profit = before_tax_profit * (1 - tax_rate)
                cash_flow = after_tax_profit + depreciation
                cash_flows.append(cash_flow)
            
            cash_flow_ls.append(cash_flows)
            # Calculate NPV
            npv = -fixed_cost + sum(cf / ((1 + discount_rate) ** (year + 1)) for year, cf in enumerate(cash_flows))
            npvs.append(npv)
        # print(np.mean(cash_flow_ls, axis=0))
        return npvs

    # Run the simulation
    npvs = simulate_npv()
    mean_npv = np.mean(npvs)
    std_npv = np.std(npvs)
    print(f"Mean NPV: {mean_npv}")
    print(f"Standard Deviation of NPV: {std_npv}")

    return {"npv": npvs}

if __name__ == "__main__":
    res = run_simulation()
    print(res)