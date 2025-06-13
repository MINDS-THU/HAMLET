import random
import json
import yaml
import importlib
from pathlib import Path
from pprint import pprint
import os
import base64
import argparse
from tqdm import tqdm
from threading import Lock   # NEW ← add near other imports at top
from concurrent.futures import ProcessPoolExecutor, as_completed, CancelledError, TimeoutError
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('./.env')

import sys
import io
import contextlib

from itertools import product   # NEW – keep with other imports

MODELS = [
    "google/gemma-3-27b-it",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "Qwen/Qwen3-32B",
    "gpt-4.1",
    "o3",
]

SKIP_PAIRS = {
    "google/gemma-3-27b-it::google/gemma-3-27b-it",
    "google/gemma-3-27b-it::mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "google/gemma-3-27b-it::Qwen/Qwen3-32B",
    "google/gemma-3-27b-it::gpt-4.1",
    "google/gemma-3-27b-it::o3",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503::google/gemma-3-27b-it", 
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503::mistralai/Mistral-Small-3.1-24B-Instruct-2503", 
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503::Qwen/Qwen3-32B", 
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503::gpt-4.1",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503::o3",
    "Qwen/Qwen3-32B::google/gemma-3-27b-it",
    "Qwen/Qwen3-32B::mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "Qwen/Qwen3-32B::Qwen/Qwen3-32B",
    "Qwen/Qwen3-32B::gpt-4.1",
    # …etc – copy the rest of your original list …
}

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# -----------------------------------------------------------------------------
# OpenTelemetry / Langfuse instrumentation (runs in every process)
# -----------------------------------------------------------------------------
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY_BARGAINING", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY_BARGAINING", "")
LANGFUSE_AUTH = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry import trace

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

trace.set_tracer_provider(trace_provider)

from src.custom_smolagents_instrumentor import CustomSmolagentsInstrumentor
from openinference.instrumentation import using_session

from smolagents import LiteLLMModel, InferenceClientModel, LogLevel, TransformersModel
from src.base_agent import CodeAgent
from apps.bargaining.tool_lib.bargaining_simulation.bargaining_simulation_tools import (
    EventManager,
    make_offer,
    respond_to_offer,
    send_message,
    wait_for_response,
    wait_for_time_period,
    quit_negotiation,
    SearchPrice,
)
from apps.bargaining.tool_lib.bargaining_simulation.utils import (
    negotiation_sanity_checks,
    eventbatch2text,
    get_current_timestamp,
    swtich_role,
    sample_private_values_for_dataset,
)

# -----------------------------------------------------------------------------
#                        Utility helpers
# -----------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Return a filesystem‑safe string for paths / IDs (does **not** change model id)."""
    return name.replace("/", "_").replace(":", "_")

# -----------------------------------------------------------------------------
#                        Negotiation helpers (unchanged)
# -----------------------------------------------------------------------------

def start_negotiating(agents):
    """Run one complete buyer–seller negotiation."""
    terminal_tools = [
        "wait_for_response",
        "wait_for_time_period",
        "quit_negotiation",
    ]

    deal = False
    latest_offer = {}
    event_list = []  # [{"agent_name": ..., "events": ...}]

    # 1) Seller posts the initial price
    listing_price = agents["seller"].run(
        "Please first decide an initial price for the item, which will be displayed to the buyer. Return the value of initial price as the final answer.",
        reset=False,
        terminal_tools=terminal_tools,
    ).action_output

    # 2) Buyer opens negotiations
    agents["buyer"].run(
        f"The seller has proposed an initial offer of {listing_price}. Please engage in negotiation.\n Think carefully and act to maximize your utility. Do not propose offers that leads to negative utility, which is even worse than no deal",
        reset=False,
        terminal_tools=terminal_tools,
    )

    # 3) Event loop
    time, event_batch = EventManager.process_next_event()
    while event_batch is not None:
        event_text, deal, latest_offer, terminate, event_list = eventbatch2text(
            time, event_batch, deal, latest_offer, event_list
        )
        if event_batch[0].event_type == "wait_for_time_period":
            assert len(event_batch) == 1
            agents[event_batch[0].agent_name].run(
                event_text,
                reset=False,
                terminal_tools=terminal_tools,
            )
        else:
            agents[swtich_role(event_batch[0].agent_name)].run(
                event_text,
                reset=False,
                terminal_tools=terminal_tools,
            )
        if terminate:
            return deal, latest_offer, event_list
        time, event_batch = EventManager.process_next_event()

    return deal, latest_offer, event_list


def process_outcome(listing, event_list, deal, deal_price):
    buyer_checks, seller_checks = negotiation_sanity_checks(
        event_list, deal, deal_price
    )
    if not deal:
        buyer_utility = 0.0
        seller_utility = 0.0
    else:
        buyer_utility = round((listing["buyer_bottomline_price"] - deal_price), 2)
        seller_utility = round((deal_price - listing["seller_bottomline_price"]), 2)

    return {
        "deal": deal,
        "deal_price": deal_price,
        "rounds": len(event_list),
        "buyer_value": listing["buyer_bottomline_price"],
        "seller_cost": listing["seller_bottomline_price"],
        "buyer_utility": buyer_utility,
        "seller_utility": seller_utility,
        "buyer_checks": buyer_checks,
        "seller_checks": seller_checks,
    }


def build_model(model_name):
    if any(x in model_name for x in ["gpt", "o3", "claude", "gemini"]):
        return LiteLLMModel(model_id=model_name)
    elif any(x in model_name for x in ["Qwen", "gemma", "Mistral"]):
        return InferenceClientModel(model_id=model_name, provider="nebius")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    # return InferenceClientModel(model_id="google/gemma-3-27b-it", provider="nebius")

# -----------------------------------------------------------------------------
#               Worker executed in a **separate process**
# -----------------------------------------------------------------------------

def _run_single_listing(
    data_idx: int,
    listing: dict,
    buyer_model_name: str,
    seller_model_name: str,
    data_split: str,
    cur_date_time: str,
):
    """Function executed in a child process. It returns a single result dict."""

    # IMPORTANT: Instrumentation must happen **inside** the child process so that
    # spans are correctly captured for each negotiation.
    CustomSmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

    session_id = f"{cur_date_time}_{_sanitize(buyer_model_name)}_{_sanitize(seller_model_name)}_{data_split}_{data_idx}"

    # Limit conversation length within a listing
    listing_local = dict(listing)  # local copy
    listing_local["max_round"] = 10

    # Reset EventManager – in a separate process, this is only local state
    EventManager.reset()

    with using_session(session_id=session_id):
        # 1) Build models **inside** the child process – avoids pickling issues
        buyer_model = build_model(buyer_model_name)
        seller_model = build_model(seller_model_name)

        # 2) Load prompt template (safe – YAML parsing is fast)
        bargaining_agent_prompt_template = yaml.safe_load(
            importlib.resources.files("apps.bargaining.prompts")
            .joinpath("bargaining_agent.yaml")
            .read_text(encoding="utf-8")
        )

        # 3) Instantiate buyer / seller agents
        common_tools = [
            make_offer,
            respond_to_offer,
            send_message,
            wait_for_response,
            wait_for_time_period,
            quit_negotiation,
            SearchPrice(
                highest_price_info=listing_local["highest_price_info"],
                lowest_price_info=listing_local["lowest_price_info"],
            ),
        ]

        buyer = CodeAgent(
            name="buyer",
            prompt_templates=bargaining_agent_prompt_template,
            additional_prompt_variables=listing_local,
            tools=common_tools,
            model=buyer_model,
            verbosity_level=LogLevel.DEBUG,
            save_to_file=None,
        )

        seller = CodeAgent(
            name="seller",
            prompt_templates=bargaining_agent_prompt_template,
            additional_prompt_variables=listing_local,
            tools=common_tools,
            model=seller_model,
            verbosity_level=LogLevel.DEBUG,
            save_to_file=None,
        )

        # 4) Run the negotiation
        deal, latest_offer, event_list = start_negotiating({"buyer": buyer, "seller": seller})

        # 5) Determine deal price
        if deal:
            if "buyer" in latest_offer:
                assert "seller" not in latest_offer
                deal_price = latest_offer["buyer"]["price"]
            elif "seller" in latest_offer:
                assert "buyer" not in latest_offer
                deal_price = latest_offer["seller"]["price"]
            else:
                raise ValueError(f"Invalid latest_offer {latest_offer}")
        else:
            deal_price = None

        if not event_list:
            raise ValueError("event_list is empty, please check the dialogue history")

        # 6) Summarise outcome
        final_outcome = process_outcome(listing_local, event_list, deal, deal_price)

        return {
            "session_id": session_id,
            "final_outcome": final_outcome,
        }


# -----------------------------------------------------------------------------
#                                  CLI
# -----------------------------------------------------------------------------
def str2bool(v):
    return v.lower() in ("true", "1", "yes", "y")

def run_and_log(idx, listing, buyer_model, seller_model, data_split, cur_date_time, log_dir):
    from pathlib import Path
    import contextlib

    log_path = Path(log_dir) / f"listing_{idx:03}.log"
    with open(log_path, "w", encoding="utf-8") as log_fp, \
         contextlib.redirect_stdout(log_fp), \
         contextlib.redirect_stderr(log_fp):
        return _run_single_listing(idx, listing, buyer_model, seller_model, data_split, cur_date_time)

# --------------------------------------------------------------------------- #
#  Main entry‐point : single executor over *all* buyer × seller pairs         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default="validation")
    parser.add_argument("--mode",
                        type=str,
                        default="uniform",
                        choices=["uniform",
                                 "force_gain_from_trade",
                                 "force_no_gain_from_trade"])
    parser.add_argument("--random_seed", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--model_filter", nargs="*", default=None,
                        help="Run only models whose ID contains any of these substrings "
                             "(e.g. --model_filter gpt Qwen)")          # NEW
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Timestamp + output locations
    # ---------------------------------------------------------------------
    cur_date_time = get_current_timestamp()
    log_subdir    = Path(args.log_dir)
    log_subdir.mkdir(parents=True, exist_ok=True)

    for f in log_subdir.glob("listing_*.log"):
        f.unlink()                                                  # fresh logs

    # ---------------------------------------------------------------------
    # Load / sample dataset *once* (pair-specific resampling happens later)
    # ---------------------------------------------------------------------
    dataset_path = Path(f"./apps/bargaining/sim_datasets/{args.data_split}_data.json")
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    processed_data = json.loads(dataset_path.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------------
    # Build tasks  (every listing for every buyer × seller, optional filter)
    # ---------------------------------------------------------------------
    selected_models = [m for m in MODELS
                       if not args.model_filter
                       or any(term in m for term in args.model_filter)]

    # tasks = []
    # idx_global = 0
    # for buyer, seller in product(selected_models, selected_models):
    #     pair_id = f"{buyer}::{seller}"
    #     if pair_id in SKIP_PAIRS:
    #         continue

    #     # per-pair log directory (keeps old layout)
    #     pair_log_dir = (Path(args.log_dir)
    #                     / f"{_sanitize(buyer)}_{_sanitize(seller)}_"
    #                       f"{args.data_split}_{cur_date_time}")
    #     pair_log_dir.mkdir(parents=True, exist_ok=True)

    #     pair_data = sample_private_values_for_dataset(
    #         processed_data,
    #         random_seed=args.random_seed,
    #         mode=args.mode,
    #     )
    #     for listing in pair_data:
    #         tasks.append(
    #             (idx_global, listing, buyer, seller,
    #              args.data_split, cur_date_time, pair_log_dir)  # ← NEW ARG
    #         )
    #         idx_global += 1

    tasks = []
    for buyer, seller in product(selected_models, selected_models):
        pair_id = f"{buyer}::{seller}"
        if pair_id in SKIP_PAIRS:
            continue

        pair_log_dir = (Path(args.log_dir)
                        / f"{_sanitize(buyer)}_{_sanitize(seller)}_"
                        f"{args.data_split}_{cur_date_time}")
        pair_log_dir.mkdir(parents=True, exist_ok=True)

        pair_data = sample_private_values_for_dataset(
            processed_data,
            random_seed=args.random_seed,
            mode=args.mode,
        )

        # 🔽 enumerate to get 0-based index **per pair**
        for idx_pair, listing in enumerate(pair_data):
            tasks.append(
                (idx_pair, listing, buyer, seller,
                args.data_split, cur_date_time, pair_log_dir)
            )


    n_pairs = len(tasks) // len(processed_data)
    print(f"Submitting {len(tasks):,} listings across {n_pairs:,} model pairs.")

    # ---------------------------------------------------------------------
    # Crash-safe JSONL outputs
    # ---------------------------------------------------------------------
    output_dir = Path("./apps/bargaining/results"); output_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------------
    # Per-pair result / failure files  ➜  same layout as the old runs
    # ---------------------------------------------------------------------
    pair_files    = {}          # (buyer, seller) -> Path to *_results.jsonl
    pair_failures = {}          # (buyer, seller) -> Path to *_failures.jsonl
    pair_locks    = {}          # one Lock per pair

    def _get_paths(buyer: str, seller: str):
        """Return (results_path, failures_path, lock) for this pair."""
        key = (buyer, seller)
        if key not in pair_files:
            fname_base = (f"{cur_date_time}_{_sanitize(buyer)}_"
                        f"{_sanitize(seller)}_{args.data_split}")
            pair_files[key]    = output_dir / f"{fname_base}_results.jsonl"
            pair_failures[key] = output_dir / f"{fname_base}_failures.jsonl"

            # start each run fresh
            pair_files[key].unlink(missing_ok=True)
            pair_failures[key].unlink(missing_ok=True)

            pair_locks[key] = Lock()
        return pair_files[key], pair_failures[key], pair_locks[key]

    def _save_result(record: dict, buyer: str, seller: str) -> None:
        """Thread-safe append to the correct <buyer>_<seller> results file."""
        res_path, _, lock = _get_paths(buyer, seller)
        with lock, open(res_path, "a", encoding="utf-8") as fp:
            json.dump(record, fp)
            fp.write("\n")
            fp.flush()

    output_file        = output_dir / f"{cur_date_time}_ALLPAIRS_results.jsonl"
    failed_output_file = output_dir / f"{cur_date_time}_ALLPAIRS_failures.jsonl"
    output_file.unlink(missing_ok=True); failed_output_file.unlink(missing_ok=True)

    save_lock = Lock()

    # ---------------------------------------------------------------------
    # Run tasks in parallel – single ProcessPoolExecutor
    # ---------------------------------------------------------------------
    FUTURE_TIMEOUT = 360         # seconds per listing
    results = []
    print(f"Running {len(tasks):,} listings with {args.num_workers} workers …\n")

    executor = None
    try:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_info = {
                executor.submit(
                    run_and_log,
                    idx, listing, buyer, seller,
                    args.data_split, cur_date_time, pair_log_dir  # ← pass it
                ): (idx, buyer, seller)
                for idx, listing, buyer, seller, _, _, pair_log_dir in tasks   # ← unpack
            }
            for future in tqdm(as_completed(future_to_info),
                               total=len(future_to_info),
                               desc=f"Processing {args.data_split} data"):
                idx, buyer, seller = future_to_info[future]
                try:
                    record = future.result(timeout=FUTURE_TIMEOUT)
                    results.append(record); _save_result(record, buyer, seller)

                except TimeoutError:
                    print(f"[{buyer} | {seller} | listing {idx}] "
                          f"timed out after {FUTURE_TIMEOUT}s – skipping.")
                    fail_record = {
                        "session_id": f"{cur_date_time}_{_sanitize(buyer)}_"
                                      f"{_sanitize(seller)}_{args.data_split}_{idx}",
                        "error": f"timeout>{FUTURE_TIMEOUT}s",
                        "listing_index": idx,
                        "buyer_model": buyer,
                        "seller_model": seller,
                    }
                    res_path, fail_path, lock = _get_paths(buyer, seller)
                    with lock, open(fail_path, "a", encoding="utf-8") as fp:
                        json.dump(fail_record, fp)
                        fp.write("\n")
                        fp.flush()


                except CancelledError:
                    print(f"[{buyer} | {seller} | listing {idx}] was cancelled.")

                except Exception as exc:
                    print(f"[{buyer} | {seller} | listing {idx}] failed: {exc}")
                    fail_record = {
                        "session_id": f"{cur_date_time}_{_sanitize(buyer)}_"
                                      f"{_sanitize(seller)}_{args.data_split}_{idx}",
                        "error": str(exc),
                        "listing_index": idx,
                        "buyer_model": buyer,
                        "seller_model": seller,
                    }

                    res_path, fail_path, lock = _get_paths(buyer, seller)
                    with lock, open(fail_path, "a", encoding="utf-8") as fp:
                        json.dump(fail_record, fp)
                        fp.write("\n")
                        fp.flush()


    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt — shutting down executor …")
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        raise

    # ---------------------------------------------------------------------
    # Optional: also write full aggregated list (old behaviour)
    # ---------------------------------------------------------------------
    # final_json = output_file.with_suffix(".json")
    # with open(final_json, "w", encoding="utf-8") as fp:
    #     json.dump(results, fp, indent=4)

    # print("\nAll listings processed.")
    # print(f"• incremental results → {output_file}")
    # print(f"• aggregated list     → {final_json}\n")
