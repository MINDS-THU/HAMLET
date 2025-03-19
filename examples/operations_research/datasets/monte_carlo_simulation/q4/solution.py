import numpy as np

def run_simulation():
    # Set simulation parameters
    n_iter = 10000
    sim_period = 6.0
    warranty_period = 1.0
    device_cost = 100

    # Arrays to store simulation outcomes
    total_costs = np.zeros(n_iter)
    warranty_failures_arr = np.zeros(n_iter)
    devices_owned_arr = np.zeros(n_iter)

    # Simulation loop
    for i in range(n_iter):
        total_cost = device_cost  # initial device purchase
        warranty_failures = 0
        devices_owned = 1  # initial device count
        current_time = 0.0

        # Continue simulation until we reach 6 years of operation
        while current_time < sim_period:
            # Sample the lifetime of the current device from a Gamma(alpha=2, beta=0.5)
            lifetime = np.random.gamma(shape=2, scale=0.5)
            
            # If the current device would last past the simulation period, we're done.
            if current_time + lifetime >= sim_period:
                break
            
            # The device fails within the simulation period; update the current time.
            current_time += lifetime
            
            # Check if the failure occurred within the warranty period
            if lifetime <= warranty_period:
                # Under warranty: replacement is free.
                warranty_failures += 1
            else:
                # Outside warranty: pay for a new device.
                total_cost += device_cost
            
            # In either case, you receive a new device.
            devices_owned += 1

        # Store the outcomes of this iteration
        total_costs[i] = total_cost
        warranty_failures_arr[i] = warranty_failures
        devices_owned_arr[i] = devices_owned

    # Calculate means and standard deviations for each outcome
    mean_total_cost = np.mean(total_costs)
    std_total_cost = np.std(total_costs)

    mean_warranty_failures = np.mean(warranty_failures_arr)
    std_warranty_failures = np.std(warranty_failures_arr)

    mean_devices_owned = np.mean(devices_owned_arr)
    std_devices_owned = np.std(devices_owned_arr)

    # Print the results
    print("Total Cost: Mean = ${:.2f}, Std = ${:.2f}".format(mean_total_cost, std_total_cost))
    print("Warranty Failures: Mean = {:.2f}, Std = {:.2f}".format(mean_warranty_failures, std_warranty_failures))
    print("Devices Owned: Mean = {:.2f}, Std = {:.2f}".format(mean_devices_owned, std_devices_owned))

    return {"total cost": total_costs, "num of failures during warranty": warranty_failures_arr, "devices owned": devices_owned_arr}

if __name__ == "__main__":
    res = run_simulation()
    print(res)