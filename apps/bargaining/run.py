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
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock   # NEW ← add near other imports at top

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('./.env')

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
    if any(x in model_name for x in ["gpt", "claude", "gemini"]):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buyer_model",  type=str, default="gpt-4.1")
    parser.add_argument("--seller_model", type=str, default="gpt-4.1")
    parser.add_argument("--data_split",   type=str, default="validation")
    parser.add_argument("--gain_from_trade", type=bool, default=False)
    parser.add_argument("--random_seed",  type=int, default=30)
    parser.add_argument("--num_workers",  type=int, default=8)
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Timestamp and dataset
    # ---------------------------------------------------------------------
    cur_date_time = get_current_timestamp()

    dataset_path = Path(f"./apps/bargaining/sim_datasets/{args.data_split}_data.json")
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        processed_data = json.load(f)
    processed_data = processed_data[:100]
    updated_processed_data = sample_private_values_for_dataset(
        processed_data,
        gain_from_trade=args.gain_from_trade,
        random_seed=args.random_seed,
    )

    # ---------------------------------------------------------------------
    # Build task list
    # ---------------------------------------------------------------------
    tasks = [
        (idx, listing, args.buyer_model, args.seller_model, args.data_split, cur_date_time)
        for idx, listing in enumerate(updated_processed_data)
    ]

    # ---------------------------------------------------------------------
    # Prepare **incremental** output file + thread‑safe writer
    # ---------------------------------------------------------------------
    output_dir = Path("./apps/bargaining/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # one‑result‑per‑line JSONL for crash‑safe saving
    output_file = output_dir / (
        f"{cur_date_time}_{_sanitize(args.buyer_model)}_"
        f"{_sanitize(args.seller_model)}_{args.data_split}_results.jsonl"
    )
    output_file.unlink(missing_ok=True)  # start fresh
    save_lock = Lock()

    def _save_result(record: dict) -> None:
        """Append one completed listing outcome (thread‑safe)."""
        with save_lock:
            with open(output_file, "a", encoding="utf-8") as fp:
                json.dump(record, fp)
                fp.write("\n")   # JSONL = one JSON per line
                fp.flush()

    # ---------------------------------------------------------------------
    # Run listings in parallel
    # ---------------------------------------------------------------------
    results = []    # keep in RAM if you still want the full list
    print(f"Running {len(tasks)} listings with {args.num_workers} workers…\n")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_idx = {executor.submit(_run_single_listing, *t): t[0] for t in tasks}

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc=f"Processing {args.data_split} data",
        ):
            idx = future_to_idx[future]
            try:
                record = future.result()   # one listing’s result
                results.append(record)     # (optional) keep aggregate
                _save_result(record)       # ← immediate, crash‑safe write
            except Exception as exc:
                print(f"[Listing {idx}] failed with: {exc}")

    # ---------------------------------------------------------------------
    # Optional: also write full aggregated list (old behaviour)
    # ---------------------------------------------------------------------
    # final_json = output_file.with_suffix(".json")
    # with open(final_json, "w", encoding="utf-8") as fp:
    #     json.dump(results, fp, indent=4)

    # print("\nAll listings processed.")
    # print(f"• incremental results → {output_file}")
    # print(f"• aggregated list     → {final_json}\n")
