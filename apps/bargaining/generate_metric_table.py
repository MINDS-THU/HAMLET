import os
import json
from collections import defaultdict
import statistics
import argparse

def compute_negotiation_metrics(
    filename,                       # <- unchanged
    *,
    gft_filter: str = "all"         # "all" | "gain" | "no-gain"
):
    buyer_utilities = []
    seller_utilities = []
    buyer_utilities_deal = []
    seller_utilities_deal = []
    buyer_violations = []
    seller_violations = []
    rounds_list = []
    deal_count = 0
    total_sessions = 0

    def _keep(entry) -> bool:
        """
        Decide whether to keep this entry under the requested gain-from-trade filter.
        """
        if gft_filter == "all":
            return True

        bv  = entry["final_outcome"]["buyer_value"]
        sc  = entry["final_outcome"]["seller_cost"]
        gft = bv > sc          # strictly positive surplus

        if gft_filter == "gain":
            return gft
        if gft_filter == "no-gain":
            return not gft
        raise ValueError(f"Unknown gft_filter: {gft_filter}")


    with open(filename, 'r') as file:

        for line in file:
            if not line.strip():
                continue  # Skip empty lines
            entry = json.loads(line)
            # ─── NEW ───
            if not _keep(entry):
                continue
            # ───────────
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
        "Total entries": total_sessions,
        "Average utility - Buyer": statistics.mean(buyer_utilities) if buyer_utilities else 0,
        "Average utility - Seller": statistics.mean(seller_utilities) if seller_utilities else 0,
        "Average utility (deal only) - Buyer": statistics.mean(buyer_utilities_deal) if buyer_utilities_deal else 0,
        "Average utility (deal only) - Seller": statistics.mean(seller_utilities_deal) if seller_utilities_deal else 0,
        "Average rule violations per negotiation - Buyer": statistics.mean(buyer_violations) if buyer_violations else 0,
        "Average rule violations per negotiation - Seller": statistics.mean(seller_violations) if seller_violations else 0,
        "Deal rate": deal_count / total_sessions if total_sessions else 0,
        "Mean number of rounds": statistics.mean(rounds_list) if rounds_list else 0
    }

parser = argparse.ArgumentParser(
    description="Aggregate bargaining simulation metrics."
)
parser.add_argument(
    "--results_dir", default="results",
    help="Directory that contains *_results.jsonl files (default: results/)"
)
parser.add_argument(
    "--gft_filter", choices=["all", "gain", "no-gain"], default="all",
    help="Filter listings by gain-from-trade condition "
         "(buyer_value > seller_cost)."
)
parser.add_argument(
    "--quiet", action="store_true",
    help="Suppress per-file progress messages."
)
args = parser.parse_args()
RESULTS_DIR = args.results_dir

mode_name = {
    "all":     "All Listings",
    "gain":    "Gain-from-Trade Only",
    "no-gain": "No-Gain Listings",
}[args.gft_filter]

# Canonical model names and their possible filename identifiers
MODEL_MAP = {
    "gpt-4.1": ["gpt-4.1"],
    "o3": ["o3"],
    "Qwen3-32B": ["Qwen", "Qwen3-32B"],
    "Gemma-3-27B": ["gemma", "Gemma", "google_gemma-3-27b-it"],
    "Mistral-3.1-24B": ["Mistral", "mistral", "Mistral-Small-3.1-24B-Instruct"]
}

MODEL_NAMES = list(MODEL_MAP.keys())

# Identify canonical model name from a filename part
def match_model_name(part):
    for canonical, patterns in MODEL_MAP.items():
        for pattern in patterns:
            if pattern in part:
                return canonical
    return None

# Metrics table: buyer_model × seller_model → metrics dict
table = defaultdict(dict)

# Scan all JSONL result files
for filename in os.listdir(RESULTS_DIR):
    if not filename.endswith("validation_results.jsonl"):
        continue

    parts = filename.split("_")
    matched_models = []

    for part in parts:
        model_name = match_model_name(part)
        if model_name and model_name not in matched_models:
            matched_models.append(model_name)

    # Match buyer/seller pairing
    if len(matched_models) == 2:
        buyer_model, seller_model = matched_models
    elif len(matched_models) == 1:
        buyer_model = seller_model = matched_models[0]  # self-play
    else:
        print(f"Skipping: {filename} — could not identify buyer/seller models")
        continue

    filepath = os.path.join(RESULTS_DIR, filename)
    print(f"Reading: {filename} → Buyer: {buyer_model}, Seller: {seller_model}")
    
    metrics = compute_negotiation_metrics(filepath, gft_filter=args.gft_filter)
    table[buyer_model][seller_model] = metrics
    if not args.quiet:                                
        print(f"  ↳ kept {metrics['Total entries']} entries "
            f"after gft_filter = {args.gft_filter}")
# Print summary tables
# Column width setting for alignment
COL_WIDTH = 20

print(f"\n=== Summary Table ({mode_name}) ===")

# Print combined utility table
def print_combined_metric_table(buyer_key, seller_key, title):
    print(f"\nMetric: {title}")
    header = "".ljust(COL_WIDTH) + "".join(name.ljust(COL_WIDTH) for name in MODEL_NAMES)
    print(header)

    for buyer in MODEL_NAMES:
        row_str = buyer.ljust(COL_WIDTH)
        for seller in MODEL_NAMES:
            result = table.get(buyer, {}).get(seller)
            if result:
                b_val = f"{result[buyer_key]:.1f}"
                s_val = f"{result[seller_key]:.1f}"
                cell = f"({b_val}, {s_val})"
            else:
                cell = "–"
            row_str += cell.ljust(COL_WIDTH)
        print(row_str)

# Print buyer/seller utility
print_combined_metric_table(
    "Average utility - Buyer", "Average utility - Seller", "Average Utility"
)

print_combined_metric_table(
    "Average utility (deal only) - Buyer", "Average utility (deal only) - Seller",
    "Average Utility on Deal"
)

# Print rule violations as combined (buyer, seller) table
print_combined_metric_table(
    "Average rule violations per negotiation - Buyer",
    "Average rule violations per negotiation - Seller",
    "Average Rule Violations"
)

# Now print the remaining scalar metrics individually
metric_keys = [
    "Deal rate",
    "Mean number of rounds"
]

for metric_key in metric_keys:
    print(f"\nMetric: {metric_key}")
    header = "".ljust(COL_WIDTH) + "".join(name.ljust(COL_WIDTH) for name in MODEL_NAMES)
    print(header)

    for buyer in MODEL_NAMES:
        row_str = buyer.ljust(COL_WIDTH)
        for seller in MODEL_NAMES:
            result = table.get(buyer, {}).get(seller)
            cell = f"{result[metric_key]:.2f}" if result else "–"
            row_str += cell.ljust(COL_WIDTH)
        print(row_str)
