import json
import statistics
import argparse

def compute_negotiation_metrics(filename):
    buyer_utilities = []
    seller_utilities = []
    buyer_utilities_deal = []
    seller_utilities_deal = []
    buyer_violations = []
    seller_violations = []
    rounds_list = []
    deal_count = 0
    total_sessions = 0

    with open(filename, 'r') as file:
        for line in file:
            if not line.strip():
                continue  # Skip empty lines
            entry = json.loads(line)
            outcome = entry["final_outcome"]
            total_sessions += 1

            buyer_utilities.append(outcome["buyer_utility"])
            seller_utilities.append(outcome["seller_utility"])
            rounds_list.append(outcome["rounds"])
            buyer_violations.append(sum(outcome["buyer_checks"].values()))
            seller_violations.append(sum(outcome["seller_checks"].values()))

            if outcome["deal"]:
                deal_count += 1
                buyer_utilities_deal.append(outcome["buyer_utility"])
                seller_utilities_deal.append(outcome["seller_utility"])

    return {
        "Average utility - Buyer": statistics.mean(buyer_utilities) if buyer_utilities else 0,
        "Average utility - Seller": statistics.mean(seller_utilities) if seller_utilities else 0,
        "Average utility (deal only) - Buyer": statistics.mean(buyer_utilities_deal) if buyer_utilities_deal else 0,
        "Average utility (deal only) - Seller": statistics.mean(seller_utilities_deal) if seller_utilities_deal else 0,
        "Average rule violations per negotiation - Buyer": statistics.mean(buyer_violations) if buyer_violations else 0,
        "Average rule violations per negotiation - Seller": statistics.mean(seller_violations) if seller_violations else 0,
        "Deal rate": deal_count / total_sessions if total_sessions else 0,
        "Mean number of rounds": statistics.mean(rounds_list) if rounds_list else 0
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute negotiation metrics from a JSONL file.")
    parser.add_argument("filename", type=str, help="Path to the JSONL file containing negotiation results")
    args = parser.parse_args()

    metrics = compute_negotiation_metrics(args.filename)
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")
