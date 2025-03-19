"""
The problem involves simulating the production process of a drug to determine the optimal start date for production to meet a delivery deadline. The simulation considers three sources of uncertainty: the time to produce a batch, the yield from each batch, and the probability of passing inspection. The goal is to ensure a high probability of meeting the delivery date by simulating the production process multiple times and analyzing the results. The simulation will involve generating random values for the production time, yield, and inspection outcome for each batch, and calculating the cumulative yield and total production time until the required quantity is met. The results will be used to determine the optimal start date for production.
"""

import numpy as np
import pandas as pd

def run_simulation():
    # Constants
    Ounces_required = 8000
    Due_date = pd.to_datetime('2013-12-01')
    Probability_pass = 0.8

    # Discrete distribution for days to produce a batch
    production_days = [5, 6, 7, 8, 9, 10, 11]
    production_probabilities = [0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05]

    # Triangular distribution for yield
    yield_min = 600
    yield_most_likely = 1000
    yield_max = 1100

    # Simulation parameters
    num_simulations = 10000

    # Results storage
    days_to_complete_list = []
    batches_required_list = []

    # Simulation
    for _ in range(num_simulations):
        cumulative_yield = 0
        total_days = 0
        batches_required = 0
        
        while cumulative_yield < Ounces_required:
            # Simulate production days
            days = np.random.choice(production_days, p=production_probabilities)
            total_days += days
            
            # Simulate yield
            yield_amount = np.random.triangular(yield_min, yield_most_likely, yield_max)
            
            # Simulate inspection
            if np.random.rand() < Probability_pass:
                cumulative_yield += yield_amount
            
            batches_required += 1
        
        days_to_complete_list.append(total_days)
        batches_required_list.append(batches_required)

    # Calculate statistics
    avg_days_to_complete = np.mean(days_to_complete_list)
    min_days_to_complete = np.min(days_to_complete_list)
    max_days_to_complete = np.max(days_to_complete_list)
    perc_5_days_to_complete = np.percentile(days_to_complete_list, 5)
    perc_95_days_to_complete = np.percentile(days_to_complete_list, 95)

    # Calculate start dates
    avg_start_date = Due_date - pd.to_timedelta(avg_days_to_complete, unit='D')
    min_start_date = Due_date - pd.to_timedelta(min_days_to_complete, unit='D')
    max_start_date = Due_date - pd.to_timedelta(max_days_to_complete, unit='D')
    perc_5_start_date = Due_date - pd.to_timedelta(perc_5_days_to_complete, unit='D')
    perc_95_start_date = Due_date - pd.to_timedelta(perc_95_days_to_complete, unit='D')

    # Print results
    print(f"Average days to complete: {avg_days_to_complete}")
    print(f"Minimum days to complete: {min_days_to_complete}")
    print(f"Maximum days to complete: {max_days_to_complete}")
    print(f"5th percentile days to complete: {perc_5_days_to_complete}")
    print(f"95th percentile days to complete: {perc_95_days_to_complete}")

    print(f"Average start date: {avg_start_date}")
    print(f"Minimum start date: {min_start_date}")
    print(f"Maximum start date: {max_start_date}")
    print(f"5th percentile start date: {perc_5_start_date}")
    print(f"95th percentile start date: {perc_95_start_date}")

    # Probability of meeting due date if starting on a specific date
    specific_start_date = pd.to_datetime('2013-07-15')
    days_from_specific_start = (Due_date - specific_start_date).days
    probability_meeting_due_date = np.mean(np.array(days_to_complete_list) <= days_from_specific_start)
    print(f"Probability of meeting due date if starting on {specific_start_date.date()}: {probability_meeting_due_date}")

    return {"days to complete": days_to_complete_list}
    # return [[np.mean(days_to_complete_list), np.std(days_to_complete_list)]]

if __name__ == '__main__':
    results = run_simulation()
    print(results)