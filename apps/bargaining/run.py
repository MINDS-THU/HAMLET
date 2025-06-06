import random
import json
import yaml
import importlib
from pathlib import Path
from pprint import pprint
import os
import base64
import argparse

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('./.env')

import os
import base64

# Set Langfuse OTEL environment variables
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY_BARGAINING", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY_BARGAINING", "")
LANGFUSE_AUTH = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

# Setup OpenTelemetry tracing
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

# Import your custom instrumentor
from src.custom_smolagents_instrumentor import CustomSmolagentsInstrumentor
from openinference.instrumentation import using_session

from smolagents import LiteLLMModel, InferenceClientModel, LogLevel, TransformersModel
from src.base_agent import CodeAgent
from apps.bargaining.tool_lib.bargaining_simulation.bargaining_simulation_tools import EventManager, make_offer, respond_to_offer, send_message, wait_for_response, wait_for_time_period, quit_negotiation, SearchPrice
from apps.bargaining.tool_lib.bargaining_simulation.utils import negotiation_sanity_checks, eventbatch2text, get_current_timestamp, swtich_role

def start_negotiating(agents):
    """
    Start the negotiation process between buyer and seller agents.
    Args:
        agents (dict): A dictionary containing the buyer and seller agents.
    Returns:
        deal (bool): Whether a deal was reached.
        latest_offer (dict): The latest offer made by either agent.
        event_list (list): A list of events that occurred during the negotiation.
    """
    terminal_tools = ['wait_for_response', 'wait_for_time_period', 'quit_negotiation']
    deal = False
    latest_offer = {}
    event_list = [] # [{"agent_name": ..., "events": ...}]

    listing_price = agents["seller"].run(
        # "Please first decide a listing price for the item, which will be displayed to the buyer. Return the value of listing price as the final answer.",
        "Please first decide an initial price for the item, which will be displayed to the buyer. Return the value of initial price as the final answer.",
        reset=False,
        terminal_tools=terminal_tools,
        ).action_output

    agents["buyer"].run(
        # f"The seller has setup a listing price of {listing_price}. Please initiate negotiation.\n Think carefully and act to maximize your utility. Do not propose offers that leads to negative utility, which is even worse than no deal.",
        f"The seller has proposed an initial offer of {listing_price}. Please engage in negotiation.\n Think carefully and act to maximize your utility. Do not propose offers that leads to negative utility, which is even worse than no deal",
        reset=False,
        terminal_tools=terminal_tools,
        )
    
    time, event_batch = EventManager.process_next_event() # the list event_batch could contain multiple events
    while event_batch is not None:
        event_text, deal, latest_offer, terminate, event_list = eventbatch2text(time, event_batch, deal, latest_offer, event_list)
        # event contains 'scheduled_time', 'event_type', 'agent_name', 'event_content'
        # event_type can be 'make_offer', 'respond_to_offer', 'send_message', 'wait_for_response', 'wait_for_time_period', 'quit_negotiation'
        if event_batch[0].event_type == "wait_for_time_period":
            assert len(event_batch) == 1
            agents[event_batch[0].agent_name].run(
                event_text,
                # max_steps=5,
                reset=False,
                terminal_tools=terminal_tools
            )
        else:
            agents[swtich_role(event_batch[0].agent_name)].run(
                event_text,
                # max_steps=5,
                reset=False,
                terminal_tools=terminal_tools
            )
        if terminate:
            return deal, latest_offer, event_list
        time, event_batch = EventManager.process_next_event() # the list event_batch could contain multiple events

    return deal, latest_offer, event_list

def process_outcome(listing, event_list, deal, deal_price):
    buyer_checks, seller_checks = negotiation_sanity_checks(event_list, deal, deal_price)
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
    if "gpt" in model_name or "o3" in model_name:
        # "gpt-4o"
        return LiteLLMModel(model_id=model_name)
    elif "Qwen" in model_name:
        # "Qwen/Qwen3-32B"
        # "Qwen/Qwen3-30B-A3B"
        return InferenceClientModel(
            model_id=model_name,
            # provider="nebius",
            # provider="novita",
            )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buyer_model", type=str, default="gpt-4.1", help="The model to use for the buyer agent.")
    parser.add_argument("--seller_model", type=str, default="gpt-4.1", help="The model to use for the seller agent.")
    parser.add_argument("--data_split", type=str, default="validation", help="The data split to use for the experiment. Options are 'training', 'validation', 'testing'.")
    args = parser.parse_args()

    cur_date_time = get_current_timestamp()

    CustomSmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

    # load dataset
    try:
        # Load the processed data from the JSON file
        with open(f"./apps/bargaining/sim_datasets/{args.data_spplit}_data.json", "r") as f:
            processed_data = json.load(f)
    except FileNotFoundError:
        print(f"./apps/bargaining/sim_datasets/{args.data_spplit}_data.json")
        exit(1)

    output_dir = "./apps/bargaining/results/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    output_file = f"{output_dir}{cur_date_time}_{args.buyer_model}_{args.seller_model}_{args.data_split}_results.json"
    
    results = []

    # iterate through the dataset
    for data_idx, listing in enumerate(processed_data):
        print("==== Data Split: {}, Listing No: {} ====".format(args.data_split, data_idx))
        session_id = f"{cur_date_time}_{args.buyer_model}_{args.seller_model}_{args.data_split}_{data_idx}"
        # reset event manager for each experiment
        EventManager.reset()
        with using_session(session_id=session_id):
            # for usage of using_session and using_user, 
            # please refer to the documentation: https://docs.arize.com/arize/observe/sessions-and-users
            # setup llm backend
            buyer_model = build_model(args.buyer_model)
            seller_model = build_model(args.seller_model)

            # load bargaining agent prompt
            bargaining_agent_prompt_template = yaml.safe_load(
                        importlib.resources.files("apps.bargaining.prompts").joinpath("bargaining_agent.yaml").read_text(encoding="utf-8")
                    )
            # instantiate buyer and seller agents
            buyer = CodeAgent(
                name='buyer', 
                prompt_templates=bargaining_agent_prompt_template,
                additional_prompt_variables=listing, 
                tools=[
                    make_offer, 
                    respond_to_offer, 
                    send_message, 
                    wait_for_response, 
                    wait_for_time_period, 
                    quit_negotiation, 
                    SearchPrice(
                        highest_price_info=listing['highest_price_info'], 
                        lowest_price_info=listing['lowest_price_info']
                        )
                        ], 
                model=buyer_model, 
                verbosity_level=LogLevel.DEBUG, 
                save_to_file=None
                )

            seller = CodeAgent(
                name='seller', 
                prompt_templates=bargaining_agent_prompt_template,
                additional_prompt_variables=listing, 
                tools=[
                    make_offer, 
                    respond_to_offer, 
                    send_message, 
                    wait_for_response, 
                    wait_for_time_period, 
                    quit_negotiation, 
                    SearchPrice(
                        highest_price_info=listing['highest_price_info'], 
                        lowest_price_info=listing['lowest_price_info']
                        )
                        ], 
                model=seller_model, 
                verbosity_level=LogLevel.DEBUG, 
                save_to_file=None
                )

            deal, latest_offer, event_list = start_negotiating({"buyer":buyer, "seller":seller})
            if deal:
                if "buyer" in latest_offer:
                    assert "seller" not in latest_offer
                    deal_price = latest_offer["buyer"]['price']
                elif "seller" in latest_offer:
                    assert "buyer" not in latest_offer
                    deal_price = latest_offer["seller"]['price']
                else:
                    deal_price = None
                    raise ValueError("Invalid lastest_offer {}".format(latest_offer))
            else:
                deal_price = None
            if event_list == []:
                raise ValueError("event_list is empty, please check the dialogue history")

            final_outcome = process_outcome(listing, event_list, deal, deal_price)
            pprint(final_outcome)

            # Append the outcome to the results list
            results.append({
                "session_id": session_id,
                "final_outcome": final_outcome
            })

            # Write the updated results to the JSON file after each session
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {output_file}")