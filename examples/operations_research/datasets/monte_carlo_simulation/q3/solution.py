import numpy as np
import pandas as pd

def run_simulation():
    # Constants
    SEAT_CAPACITY = 16
    TICKET_PRICE = 225
    FIXED_COST = 900
    PASSENGER_COST = 100
    COMPENSATION_COST = TICKET_PRICE  # Compensation for bumped passengers
    NO_SHOW_PROB = 0.04
    SIMULATIONS = 10000

    # Function to simulate profit
    def simulate_profit(reservation_count):
        profits = []
        
        for _ in range(SIMULATIONS):
            # Simulate the number of passengers who show up
            passengers_showing_up = np.random.binomial(reservation_count, 1 - NO_SHOW_PROB)
            
            # Calculate revenue and costs
            revenue = reservation_count * TICKET_PRICE
            passenger_variable_cost = passengers_showing_up * PASSENGER_COST
            total_cost = FIXED_COST + passenger_variable_cost
            
            # Compensation for bumped passengers
            if passengers_showing_up > SEAT_CAPACITY:
                bumped_passengers = passengers_showing_up - SEAT_CAPACITY
                compensation = bumped_passengers * COMPENSATION_COST
            else:
                compensation = 0
            
            # Compute final profit
            profit = revenue - total_cost - compensation
            profits.append(profit)
        
        return np.array(profits)

    # Simulate for reservation numbers 16 to 20
    results = {}
    for reservations in range(16, 21):
        profits = simulate_profit(reservations)
        results[f"profit for selling {reservations} reservations"] = profits

    return results

if __name__ == "__main__":
    res = run_simulation()
    print(res)