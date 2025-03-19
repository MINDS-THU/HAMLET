"""
The simulation models the cash‐flow of a new car project over ten years. For years 1–5, unit sales are generated using a triangular distribution with parameters (min=50000, mode=75000, max=85000). In each subsequent year from 2 to 5, sales decline by a factor of (1 – d) where d is a randomly generated decay rate (triangular with min=0.05, mode=0.08, max=0.1). Unit contribution starts at 4000 in year 1 and decays each year by a fixed rate of 0.04. Depreciation is taken evenly during years 1–5 (fixed_development_cost/5). For years 6–10 the car continues to be sold provided that a discontinuation check passes: at the beginning of year 6 the simulation examines whether year‐5 sales are below a cutoff (45000). If so, there is a 50% chance that sales are discontinued (and remain zero) for future years. Similarly, for subsequent years (years 7–10) if sales in the previous active year are below 45000, then with 50% probability the sales stop. Cash flow in years 1–5 is calculated as after‐tax profit plus depreciation (with tax rate 0.4 applied to the before‐tax profit), while in years 6–10, with no depreciation, the after‐tax profit (based solely on revenue) is used. The project net present value (NPV) is computed using a discount rate of 0.1 over all ten years and then reduced by the total fixed development cost (700000000). Also, the simulation counts the number of years in which sales occurred (i.e. positive sales). Finally, a Monte Carlo simulation is run (for example 10000 iterations) and the summary statistics (minimum, mean, maximum and standard deviation) for NPV and years of sales are printed for comparison with the Excel Output Results tab. The code below implements these steps in Python.
"""

import random
import math
import statistics

def run_simulation():
    # Parameters from the Excel simulation
    fixed_development_cost = 700000000
    year1_contribution = 4000
    annual_contribution_decay = 0.04

    a1_sales_min = 50000
    a1_sales_mode = 75000
    a1_sales_max = 85000

    decay_min = 0.05
    decay_mode = 0.08
    decay_max = 0.1

    tax_rate = 0.4

    discount_rate = 0.1

    sales_cutoff = 45000
    prob_discontinue = 0.5

    num_years = 10
    num_simulations = 10000

    # Function to draw a triangular random number
    def draw_triangular(low, mode, high):
        # Python's random.triangular accepts (low, high, mode)
        return random.triangular(low, high, mode)

    # Function to compute NPV from a list of annual cashflows
    # Cashflow at year i is discounted by (1+discount_rate)^i, i starting at 1
    def compute_npv(cashflows, discount_rate):
        npv = 0
        for i, cf in enumerate(cashflows, start=1):
            npv += cf / ((1 + discount_rate) ** i)
        return npv

    # Run the Monte Carlo simulation
    npv_results = []
    years_sales_results = []

    for sim in range(num_simulations):
        annual_sales = [0] * num_years   # store unit sales per year
        unit_contributions = [0] * num_years  # store unit contribution per year
        cashflows = [0] * num_years
        discontinuation_flags = [False] * num_years  # True if discontinued at that year onward

        # Year 1 (always active)
        # Draw initial unit sales from a triangular distribution
        annual_sales[0] = draw_triangular(a1_sales_min, a1_sales_mode, a1_sales_max)
        # Unit contribution in year1
        unit_contributions[0] = year1_contribution
        
        # Years 1-5: apply depreciation and decline in sales
        # Depreciation is fixed in each of year 1-5
        depreciation = fixed_development_cost / 5

        for year in range(1, 5):  # years 2 to 5 (indices 1 to 4)
            # Sales decline by a factor (1 - decay rate) using triangular decay rate
            decay_rate = draw_triangular(decay_min, decay_mode, decay_max)
            annual_sales[year] = annual_sales[year-1] * (1 - decay_rate)
            # Unit contribution decays by fixed rate each year
            unit_contributions[year] = unit_contributions[year-1] * (1 - annual_contribution_decay)

        # Determine discontinuation for year 6 based on year 5 sales (index 4)
        active = True
        if annual_sales[4] < sales_cutoff:
            if random.random() < prob_discontinue:
                active = False
        
        # Years 6 to 10: if active, compute sales and unit contribution; if not, sales remain 0
        for year in range(5, num_years):  # years 6 to 10 (indices 5 to 9)
            if not active:
                annual_sales[year] = 0
                # We can leave unit contribution as 0 (won't affect cash flow)
                unit_contributions[year] = 0
            else:
                # For sales: decline from previous year's sales with decaying factor
                decay_rate = draw_triangular(decay_min, decay_mode, decay_max)
                annual_sales[year] = annual_sales[year-1] * (1 - decay_rate)
                # Unit contribution continues to decay by a fixed rate
                unit_contributions[year] = unit_contributions[year-1] * (1 - annual_contribution_decay)
                
                # Check for discontinuation for the next period if this year's sales drop below cutoff
                if annual_sales[year] < sales_cutoff:
                    if random.random() < prob_discontinue:
                        active = False

        # Compute cash flows for each year
        # For years 1-5, cash flow = (revenue - depreciation after tax) + depreciation
        # Revenue = unit_sales * unit_contribution
        for year in range(5):  # years 1 to 5
            revenue = annual_sales[year] * unit_contributions[year]
            before_tax = revenue - depreciation
            after_tax = before_tax * (1 - tax_rate)
            cashflows[year] = after_tax + depreciation
        
        # For years 6-10, there is no depreciation: cash flow = revenue*(1 - tax_rate)
        for year in range(5, num_years):
            revenue = annual_sales[year] * unit_contributions[year]
            before_tax = revenue
            after_tax = before_tax * (1 - tax_rate)
            cashflows[year] = after_tax

        # Compute NPV over 10 years and subtract the fixed development cost
        npv = compute_npv(cashflows, discount_rate) - fixed_development_cost
        npv_results.append(npv)
        
        # Count the number of years in which sales were positive
        years_with_sales = sum(1 for s in annual_sales if s > 0)
        years_sales_results.append(years_with_sales)

    # Calculate summary statistics for NPV
    npv_min = min(npv_results)
    npv_mean = statistics.mean(npv_results)
    npv_max = max(npv_results)
    npv_std = statistics.pstdev(npv_results)  # population std dev

    # Calculate summary statistics for Years of Sales
    ys_min = min(years_sales_results)
    ys_mean = statistics.mean(years_sales_results)
    ys_max = max(years_sales_results)
    ys_std = statistics.pstdev(years_sales_results)

    # Print the results in a format similar to the Output Results tab in the Excel file
    print("--- Simulation Results (based on {} runs) ---".format(num_simulations))
    print(f"NPV: Min = {npv_min:,.0f}, Mean = {npv_mean:,.0f}, Max = {npv_max:,.0f}, Std Dev = {npv_std:,.0f}")
    print(f"Years of Sales: Min = {ys_min}, Mean = {ys_mean:.3f}, Max = {ys_max}, Std Dev = {ys_std:.6f}")

    # The printed values should be comparable to the Excel Output Results:
    # NPV: Min ~ -72,332,880, Mean ~ 155,363,800, Max ~ 363,054,500, Std Dev ~ 98,916,500
    # Years of Sales: Min ~ 6, Mean ~ 7.983, Max ~ 10, Std Dev ~ 1.344812

    return {"npv":npv_results, "years of sales": years_sales_results}

if __name__ == "__main__":
    res = run_simulation()
    print(res)
