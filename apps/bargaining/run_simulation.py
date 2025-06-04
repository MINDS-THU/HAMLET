import random
import json
import yaml
import importlib
from pathlib import Path
from pprint import pprint
import os
import base64

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('./.env')

import os
import base64
 
# ✅ Set Langfuse OTEL environment variables
LANGFUSE_PUBLIC_KEY = "pk-lf-c2f391d5-7aef-4827-9f20-69a624796ee2"
LANGFUSE_SECRET_KEY = "sk-lf-b5339d7d-a627-45de-b4b0-03d7c0da3ad2"
LANGFUSE_AUTH = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"  # EU
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"
os.environ["HF_TOKEN"] = "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"  # Replace with yours if needed

# ✅ Setup OpenTelemetry tracing
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry import trace

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

# ✅ Import your custom instrumentor
from src.custom_smolagents_instrumentor import CustomSmolagentsInstrumentor
from openinference.instrumentation import using_session


from smolagents import LiteLLMModel, InferenceClientModel, LogLevel, TransformersModel
from src.base_agent import CodeAgent
from apps.bargaining.tool_lib.code.simulation_tools import EventManager, make_offer, respond_to_offer, send_message, wait_for_response, wait_for_time_period, quit_negotiation, SearchPrice
from apps.bargaining.tool_lib.code.utils import negotiation_sanity_checks, eventbatch2text, get_current_timestamp, swtich_role


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
    CustomSmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

    # load amazon dataset
    # num_listings = 10
    with open("./apps/bargaining/datasets/amazon/processed_data.json", "r") as file:
        processed_data = json.load(file)

    ## Parameters
    start = 0.2
    end = 0.8
    step = 0.2

    b_list = [[1.1, 1.3], [0.9, 1.1], [0.7, 0.9]]
    s_list = [[0.7, 0.9], [0.9, 1.1], [1.1, 1.3]]
    exp_results = {}
    cur_date_time = get_current_timestamp()

    # root directory for bargaining
    base_dir = Path(".") / "apps" / "bargaining"
    # Full path with timestamp and model name
    log_dir = base_dir / "agent_lib" / "history" / f"{cur_date_time}"
    # Create the directory
    os.makedirs(log_dir, exist_ok=True)

    # with tracer.start_as_current_span("Session") as span:
    #     span.set_attribute("langfuse.session.id", "session-abc-123")
    with using_session(session_id="my-session-id"):
        # setup llm backend
        model_name = "gpt-4.1"
        if "gpt" in model_name or "o1" in model_name:
            # "gpt-4o"
            model = LiteLLMModel(model_id=model_name)
        elif "claude" in model_name:
            # "anthropic/claude-3-7-sonnet-latest"
            anthropic_api_key = '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'
            model = LiteLLMModel(model_id=model_name, api_key=anthropic_api_key)
        elif "gemini" in model_name:
            # "gemini-2.5-pro-exp-03-25"
            gemini_api_key = "AIzaSyB555_-zH-i5asvNWCW6NNXCLyW-zrhxDk"
            model = LiteLLMModel(model_id=model_name, api_key=gemini_api_key)
        elif "Llama" in model_name:
            # "meta-llama/Llama-3.1-8B-Instruct"
            # Gemma
            # "Qwen/Qwen2.5-32B-Instruct"
            model = TransformersModel(model_id=model_name)
        elif "Qwen" in model_name:
            # "Qwen/Qwen3-32B"
            # "Qwen/Qwen3-30B-A3B"
            model = InferenceClientModel(
                model_id=model_name,
                # provider="nebius",
                # provider="novita",
                token="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",
                )

        # load bargaining agent prompt
        bargaining_agent_prompt_template = yaml.safe_load(
                    importlib.resources.files("apps.bargaining.prompts").joinpath("bargaining_agent.yaml").read_text(encoding="utf-8")
                )

        for buyer_fraction_range, seller_fraction_range in zip(b_list, s_list):
            # selected_listings = sample_unique_items(processed_data, num_listings)
            selected_listings = [processed_data[0]]
            outcome_list = []
            folder_name = "_".join(f"{a}-{b}" for a, b in [buyer_fraction_range, seller_fraction_range]) # 0.2-0.4_0.2-0.4
            os.makedirs(log_dir / f"{folder_name}", exist_ok=True)
            for i in range(len(selected_listings)):
                session_id = f"{cur_date_time}_" + folder_name + "_exp_{}".format(i)
                EventManager.reset()
                # reset event manager for each experiment
                # print("==== Listing No {} ====".format(i))
                listing = selected_listings[i]
                # generate buyer and seller bottomline price
                buyer_fraction = random.uniform(buyer_fraction_range[0], buyer_fraction_range[1])
                seller_fraction = random.uniform(seller_fraction_range[0], seller_fraction_range[1])
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
                    save_to_file=log_dir / f"{folder_name}" / "exp_{}.txt".format(i)
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
                    save_to_file=log_dir / f"{folder_name}" / "exp_{}.txt".format(i)
                    )

                deal, latest_offer, event_list = start_negotiating(
                    {"buyer":buyer, "seller":seller})
                print("==========================")
                print(latest_offer)
                if deal:
                    # deal_price = latest_offer['price']
                    if "buyer" in latest_offer:
                        assert "seller" not in latest_offer
                        deal_price = latest_offer["buyer"]['price']
                    elif "seller" in latest_offer:
                        assert "buyer" not in latest_offer
                        deal_price = latest_offer["seller"]['price']
                    else:
                        print("Invalid lastest_offer {}".format(latest_offer))
                        deal_price = None
                        continue
                else:
                    deal_price = None
                if event_list == []:
                    continue
                # compute summary statistics of the outcome
                final_outcome = process_outcome(listing, buyer_bottomline_price, seller_bottomline_price, event_list, deal, deal_price)
                pprint(final_outcome)
                outcome_list.append(
                    {
                        "deal": final_outcome["deal"],
                        "rounds": len(final_outcome["event_list"]),
                        "buyer_utility": final_outcome["buyer_utility"],
                        "seller_utility": final_outcome["seller_utility"],
                        "buyer_sanity_check_results": final_outcome["buyer_sanity_check_results"],
                        "seller_sanity_check_results": final_outcome["seller_sanity_check_results"],
                        "history_lowest_price": listing['lowest_price_info'],
                        "history_highest_price": listing['highest_price_info'],
                        "average_price": listing['average_price']
                    }
                )
                buyer.logger.save_outcome(deal=final_outcome["deal"], deal_price=deal_price, rounds=len(final_outcome["event_list"]), utility=final_outcome["buyer_utility"], sanity_checks=final_outcome["buyer_sanity_check_results"], 
                                        knowledge={"history_lowest_price": listing['lowest_price_info'], "history_highest_price": listing['highest_price_info'],"buyer_value":buyer_bottomline_price})
                seller.logger.save_outcome(deal=final_outcome["deal"], deal_price=deal_price, rounds=len(final_outcome["event_list"]), utility=final_outcome["seller_utility"], sanity_checks=final_outcome["seller_sanity_check_results"],
                                        knowledge={"history_lowest_price": listing['lowest_price_info'], "history_highest_price": listing['highest_price_info'], "seller_cost":seller_bottomline_price})

                buyer.memory.reset()
                buyer.monitor.reset()
                seller.memory.reset()
                seller.monitor.reset()
                print("====     end      ====")

            
            total_valid_exps = len(outcome_list)
            total_deals = 0.0
            total_rounds = 0.0
            total_avg_prices = 0.0
            total_avg_prices_on_deal = 0.0
            buyer_total_utility = 0.0
            seller_total_utility = 0.0
            buyer_total_proposed_worse_offer_than_rejected = 0.0
            buyer_total_accepted_worse_offer_later = 0.0
            seller_total_proposed_worse_offer_than_rejected = 0.0
            seller_total_accepted_worse_offer_later = 0.0

            for outcome in outcome_list:
                total_deals += outcome["deal"]
                total_rounds += outcome["rounds"]
                total_avg_prices += outcome["average_price"]
                if outcome["deal"]:
                    total_avg_prices_on_deal += outcome["average_price"]
                buyer_total_utility += outcome["buyer_utility"]
                seller_total_utility += outcome["seller_utility"]
                buyer_total_proposed_worse_offer_than_rejected += outcome["buyer_sanity_check_results"]["proposed_worse_offer_than_rejected"]
                seller_total_proposed_worse_offer_than_rejected += outcome["seller_sanity_check_results"]["proposed_worse_offer_than_rejected"]
                buyer_total_accepted_worse_offer_later += outcome["buyer_sanity_check_results"]["accepted_worse_offer_later"]
                seller_total_accepted_worse_offer_later += outcome["seller_sanity_check_results"]["accepted_worse_offer_later"]


            if total_deals <= 0.01:
                avg_price_on_deal = 0
            else:
                avg_price_on_deal = total_avg_prices_on_deal / total_deals
            print("summary:")
            print("total valid exps: {}".format(total_valid_exps))
            print("[avg deal] {}".format(total_deals/total_valid_exps))
            print("[avg rounds] {}".format(total_rounds/total_valid_exps))
            print("[avg price] {}".format(total_avg_prices/total_valid_exps))
            print("[avg price conditioned on deal] {}".format(avg_price_on_deal))
            print("[avg utility] buyer: {}, seller: {}".format(buyer_total_utility/total_valid_exps, seller_total_utility/total_valid_exps))
            print("[avg proposed_worse_offer_than_rejected] buyer: {}, seller: {}".format(buyer_total_proposed_worse_offer_than_rejected/total_valid_exps, seller_total_proposed_worse_offer_than_rejected/total_valid_exps))
            print("[avg accepted_worse_offer_later] buyer: {}, seller: {}".format(buyer_total_accepted_worse_offer_later/total_valid_exps, seller_total_accepted_worse_offer_later/total_valid_exps))

            exp_results[folder_name] = {
                "total_valid_exps": total_valid_exps,
                "avg_deal": total_deals / total_valid_exps,
                "avg_rounds": total_rounds / total_valid_exps,
                "avg_price": total_avg_prices / total_valid_exps,
                "avg_price_on_deal": avg_price_on_deal,
                "avg_utility": {
                    "buyer": buyer_total_utility / total_valid_exps,
                    "seller": seller_total_utility / total_valid_exps
                },
                "avg_proposed_worse_offer_than_rejected": {
                    "buyer": buyer_total_proposed_worse_offer_than_rejected / total_valid_exps,
                    "seller": seller_total_proposed_worse_offer_than_rejected / total_valid_exps
                },
                "avg_accepted_worse_offer_later": {
                    "buyer": buyer_total_accepted_worse_offer_later / total_valid_exps,
                    "seller": seller_total_accepted_worse_offer_later / total_valid_exps
                }
            }

            # Write to JSON file
            with open(log_dir / f"{folder_name}" / "exp_result.json", "w") as f:
                json.dump(exp_results[folder_name], f, indent=4)  # `indent=4` makes it pretty
        print(exp_results)