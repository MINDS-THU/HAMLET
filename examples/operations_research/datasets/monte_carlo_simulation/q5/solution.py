import numpy as np
import pandas as pd

def run_simulation():
    # Constants
    salary = 50000
    tax_rate = 0.30
    mean_medical_expense = 2000
    std_medical_expense = 500
    num_simulations = 10000  # Number of simulations

    # TSB contribution range
    tsb_contributions = np.arange(1000, 3250, 250)

    # Store results
    results = {}

    # Monte Carlo Simulation with the correct formula
    for tsb in tsb_contributions:
        medical_expenses = np.random.normal(mean_medical_expense, std_medical_expense, num_simulations)
        
        # Calculate taxable income and after-tax income
        taxable_income = salary - tsb
        after_tax_income = taxable_income * (1 - tax_rate)
        
        # Compute additional medical expenses
        additional_medical_expenses = np.maximum(0, medical_expenses - tsb)
        
        # Compute money left
        money_left = after_tax_income - additional_medical_expenses
        mean_money_left = np.mean(money_left)
        std_money_left = np.std(money_left)
        results[f"tsb contribution {tsb}"] = money_left
        # results.append((tsb, mean_money_left, std_money_left))
    return results


if __name__ == "__main__":
    res = run_simulation()
    print(res)    