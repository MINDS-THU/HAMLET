#!/usr/bin/env python3
"""
Compute bucketed negotiation metrics.

Examples
--------
# keep every listing (default)
python calculate_metrics_from_results_split_by_seller_cost.py  \
       results.jsonl sim_datasets/validation_data.json

# only listings with gain-from-trade (buyer_value > seller_cost)
python calculate_metrics_from_results_split_by_seller_cost.py  \
       results.jsonl sim_datasets/validation_data.json --gft-filter gain

# only no-gain listings (buyer_value ≤ seller_cost)
python calculate_metrics_from_results_split_by_seller_cost.py  \
       results.jsonl sim_datasets/validation_data.json --gft-filter no-gain
"""
import json, argparse, statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

# ─────────────────────────── bucket definitions ──────────────────────────────
BUCKETS: List[Tuple[float, float, str]] = [
    (0.6, 0.7, "0.6–0.7×cost"),
    (0.7, 0.8, "0.7–0.8×cost"),
    (0.8, 0.9, "0.8–0.9×cost"),
]

def new_agg():
    return dict(
        count=0, u_b=[], u_s=[], u_b_d=[], u_s_d=[],
        rv_b=[], rv_s=[], rounds=[], deals=[]
    )

def which_bucket(r: float):
    for lo, hi, name in BUCKETS:
        if lo <= r < hi:
            return name
    return None

# ───────────────────────────── helper functions ──────────────────────────────
mean = lambda xs: stats.mean(xs) if xs else 0.0

def load_lowest(path: Path) -> List[float]:
    with path.open() as fp:
        return [p["lowest_price_info"]["lowest_price"] for p in json.load(fp)]

def keep(outcome: dict, mode: str) -> bool:
    """Return True if this listing passes the gain-from-trade filter."""
    if mode == "all":
        return True
    surplus = outcome["buyer_value"] > outcome["seller_cost"]
    return surplus if mode == "gain" else not surplus

# ────────────────────────────────── main ─────────────────────────────────────
def main(results: Path, lowest_prices: Path, gft_filter: str):
    lowest = load_lowest(lowest_prices)
    buckets = defaultdict(new_agg)

    with results.open() as fp:
        for line in fp:
            if not line.strip():
                continue
            rec = json.loads(line)
            out = rec["final_outcome"]

            if not keep(out, gft_filter):
                continue

            idx = int(rec["session_id"].split("_")[-1])
            bname = which_bucket(out["seller_cost"] / lowest[idx])
            if bname is None:
                continue

            b = buckets[bname]
            b["count"] += 1
            b["u_b"].append(out["buyer_utility"])
            b["u_s"].append(out["seller_utility"])
            b["rv_b"].append(sum(out["buyer_checks"].values()))
            b["rv_s"].append(sum(out["seller_checks"].values()))
            b["rounds"].append(out["rounds"])
            b["deals"].append(1 if out["deal"] else 0)
            if out["deal"]:
                b["u_b_d"].append(out["buyer_utility"])
                b["u_s_d"].append(out["seller_utility"])

    # ───── assemble rows ─────
    header = [
        "c-bucket(n)",
        "(U_b,U_s)",
        "(U_b|deal,U_s|deal)",
        "deal rate",
        "(RV_b,RV_s)",
        "rounds",
    ]
    rows = [header]

    for _, _, name in BUCKETS:
        b = buckets[name]
        rows.append([
            f"{name} ({b['count']})",
            f"({mean(b['u_b']):.2f}, {mean(b['u_s']):.2f})",
            f"({mean(b['u_b_d']):.2f}, {mean(b['u_s_d']):.2f})",
            f"{mean(b['deals']):.2f}",
            f"({mean(b['rv_b']):.2f}, {mean(b['rv_s']):.2f})",
            f"{mean(b['rounds']):.2f}",
        ])

    # column widths → aligned printing
    col_w = [max(len(row[c]) for row in rows) for c in range(len(header))]
    for row in rows:
        print("  ".join(cell.ljust(col_w[i]) for i, cell in enumerate(row)))

# ──────────────────────────────── CLI setup ──────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print bucketed negotiation metrics (aligned)."
    )
    parser.add_argument("results",        type=Path, help="JSONL negotiation results")
    parser.add_argument("lowest_prices",  type=Path, help="JSON with lowest prices")
    parser.add_argument(
        "--gft-filter",
        choices=["all", "gain", "no-gain"],
        default="all",
        help="Listing subset: all (default), gain (buyer>cost), or no-gain",
    )
    args = parser.parse_args()
    main(args.results, args.lowest_prices, args.gft_filter)
