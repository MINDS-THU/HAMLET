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
                max_steps=5,
                reset=False,
                terminal_tools=terminal_tools
            )
        else:
            agents[swtich_role(event_batch[0].agent_name)].run(
                event_text,
                max_steps=5,
                reset=False,
                terminal_tools=terminal_tools
            )
        if terminate:
            return deal, latest_offer, event_list
        time, event_batch = EventManager.process_next_event() # the list event_batch could contain multiple events

    return deal, latest_offer, event_list

def process_outcome(listing, buyer_bottomline_price, seller_bottomline_price, event_list, deal, deal_price):
    outcome = {
        "history_lowest_price": listing['lowest_price_info'],
        "history_highest_price": listing['highest_price_info'],
        "buyer_value": buyer_bottomline_price,
        "seller_cost": seller_bottomline_price,
        "event_list": event_list,
        "deal": deal,
        "deal_price": deal_price
        }
    buyer_checks, seller_checks = negotiation_sanity_checks(outcome)
    if not deal:
        buyer_utility = 0.0
        seller_utility = 0.0
    else:
        buyer_utility = round((buyer_bottomline_price - deal_price), 2)
        seller_utility = round((deal_price - seller_bottomline_price), 2)

    outcome["buyer_sanity_check_results"] = buyer_checks
    outcome["seller_sanity_check_results"] = seller_checks
    outcome["buyer_utility"] = buyer_utility
    outcome["seller_utility"] = seller_utility
    return outcome

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buyer_fraction", type=float, default=1.2, help="Fraction of buyer's willingness to pay")
    parser.add_argument("--seller_fraction", type=float, default=0.8, help="Fraction of seller's cost base")
    parser.add_argument("--dataset", type=str, default="amazon", help="amazon or craigslist")
    parser.add_argument("--data_idx", type=int, default=0, help="0-929 for amazon")
    args = parser.parse_args()
    cur_date_time = get_current_timestamp()
    session_id = f"{cur_date_time}_{args.dataset}_{args.data_idx}_bf_{str(args.buyer_fraction).replace('.', '_')}_sf_{str(args.seller_fraction).replace('.', '_')}"

    CustomSmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

    # load dataset
    try:
        with open(f"./apps/bargaining/datasets/{args.dataset}/processed_data.json", "r") as file:
            processed_data = json.load(file)
    except FileNotFoundError:
        print(f"File not found: ./apps/bargaining/datasets/{args.dataset}/processed_data.json")
        exit(1)

    with using_session(session_id=session_id):
        # for usage of using_session and using_user, 
        # please refer to the documentation: https://docs.arize.com/arize/observe/sessions-and-users
        # setup llm backend
        model_name = "gpt-4.1"
        if "gpt" in model_name or "o3" in model_name:
            # "gpt-4o"
            model = LiteLLMModel(model_id=model_name)
        elif "Llama" in model_name:
            # "meta-llama/Llama-3.1-8B-Instruct"
            model = TransformersModel(model_id=model_name)
        elif "Qwen" in model_name:
            # "Qwen/Qwen3-32B"
            # "Qwen/Qwen3-30B-A3B"
            model = InferenceClientModel(
                model_id=model_name,
                # provider="nebius",
                # provider="novita",
                )

        # load bargaining agent prompt
        bargaining_agent_prompt_template = yaml.safe_load(
                    importlib.resources.files("apps.bargaining.prompts").joinpath("bargaining_agent.yaml").read_text(encoding="utf-8")
                )
        exp_results = {}

        EventManager.reset()
        # reset event manager for each experiment
        print("==== Dataset: {}, Listing No: {} ====".format(args.dataset, args.data_idx))
        listing = processed_data[args.data_idx]
        buyer_fraction = float(args.buyer_fraction)
        seller_fraction = float(args.seller_fraction)
        buyer_bottomline_price = round(listing['average_price'] * buyer_fraction, 2)
        seller_bottomline_price = round(listing['average_price'] * seller_fraction, 2)
        listing["buyer_bottomline_price"] = buyer_bottomline_price
        listing["seller_bottomline_price"] = seller_bottomline_price
        listing["max_round"] = EventManager._max_events

        pprint(listing)

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
            model=model, 
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
            model=model, 
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
        
        # compute summary statistics of the outcome
        final_outcome = process_outcome(listing, buyer_bottomline_price, seller_bottomline_price, event_list, deal, deal_price)
        pprint(final_outcome)
        
        buyer.logger.save_outcome(deal=final_outcome["deal"], deal_price=deal_price, rounds=len(final_outcome["event_list"]), utility=final_outcome["buyer_utility"], sanity_checks=final_outcome["buyer_sanity_check_results"], 
                                knowledge={"history_lowest_price": listing['lowest_price_info'], "history_highest_price": listing['highest_price_info'],"buyer_value":buyer_bottomline_price})
        seller.logger.save_outcome(deal=final_outcome["deal"], deal_price=deal_price, rounds=len(final_outcome["event_list"]), utility=final_outcome["seller_utility"], sanity_checks=final_outcome["seller_sanity_check_results"],
                                knowledge={"history_lowest_price": listing['lowest_price_info'], "history_highest_price": listing['highest_price_info'], "seller_cost":seller_bottomline_price})