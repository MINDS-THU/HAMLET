import numpy as np
from scipy.stats import ks_2samp
import math
import os
import importlib.util

"""
use K-S test to check whether the LLM outputs and reference outputs align

output of verifier: not boolean, but percentage of correctness

oneshot 
vs try multiple times until success or fix 5 times
"""

def load_module(module_name, file_path):
    """Dynamically load a Python module from the given file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def verify_monte_carlo_results(problem_dir, llm_results, alpha=0.05):
    solution_path = os.path.join(problem_dir, "solution.py")
    if not os.path.exists(solution_path):
        raise ValueError(f"{solution_path}: missing solution.py")
        
    # Load and run the reference simulation.
    try:
        reference_module = load_module("reference_solution", solution_path)
        reference_results = reference_module.run_simulation()
    except Exception as e:
        print("Error running run_simulation() from reference solution:", e)

    # Ensure both inputs have the same set of parameters.
    if set(reference_results.keys()) != set(llm_results.keys()):
        raise ValueError("Reference and LLM results must have the same set of parameters.")

    results_summary = {}
    num_failed = 0

    for param in reference_results:
        ref_samples = reference_results[param]
        llm_samples = llm_results[param]

        stat, p_value = ks_2samp(ref_samples, llm_samples)
        if p_value <= alpha:
            num_failed += 1
        results_summary[param] = {
            "K-S": p_value > alpha # we cannot reject the null hypothesis that they are from the same distribution
        }

    total_params = len(reference_results)
    score = num_failed/total_params
    return score, {
        "Results": results_summary,
        "NumTotal": total_params,
        "NumFailed": num_failed,
    }

# Example usage:
# if __name__ == "__main__":
#     reference_results = {
#         "Parameter 1": {"mean": 50.0, "std": 10.0},
#         "Parameter 2": {"mean": 30.0, "std": 5.0},
#         "Parameter 3": {"mean": 70.0, "std": 15.0},
#     }

#     llm_results = {
#         "Parameter 1": {"mean": 49.8, "std": 10.1},
#         "Parameter 2": {"mean": 35.5, "std": 6.5},
#         "Parameter 3": {"mean": 70.5, "std": 14.8},
#     }

#     overall_verdict, verification_output = verify_monte_carlo_results(reference_results, llm_results, distance_threshold=1.0)
#     print(f"Overall verdict: {overall_verdict}")
#     import pprint
#     pprint.pprint(verification_output)
