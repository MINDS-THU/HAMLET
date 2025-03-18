import heapq, itertools
from collections import namedtuple
import random
import json
from typing import Dict
from copy import deepcopy
from pprint import pprint

from .utils import SimulationClock, sample_unique_items, negotiation_sanity_checks
from src import LiteLLMModel, LogLevel, tool, GradioUI
from src.customized_agents import BargainingAgent

#### Define a Global Event Manager ####
Event = namedtuple('Event', ['scheduled_time', 'event_type', 'agent_name', 'event_content'])

class EventManager:
    _clock = SimulationClock()  # Reference to SimulationClock, measured in seconds
    _event_queue = []
    _max_events = 50
    _events_processed = 0
    _counter = itertools.count()

    @classmethod
    def schedule(cls, event_type, agent_name, event_content):
        """
        Schedule a new event.
        
        The scheduled_time advances in two cases:
        1. The agent explicitly chooses to wait (wait_for_time_period).
        2. A default LLM response time is used for non-wait actions.
        
        Args:
            event_type (str): The type of event (e.g., make_offer, respond_to_offer, etc.).
            agent_name (str): The name of the agent.
            event_content (dict): The content associated with the event.
        """
        assert event_type in ['make_offer', 'respond_to_offer', 'send_message', 'wait_for_response', 'wait_for_time_period', 'quit_negotiation']

        # If agent wants to wait explicitly, use the provided wait_time
        if event_type == 'wait_for_response':
            return  # Do nothing; just wait for opponent response
        elif event_type == 'wait_for_time_period':
            assert 'duration' in event_content
            scheduled_time = cls._clock.now() + event_content['duration']
        else:
            # Check if the most recent event in the queue is from the same agent and not a wait event
            if cls._event_queue:
                last_scheduled_time, _, last_event = cls._event_queue[0]
                if last_event.agent_name == agent_name and last_event.event_type not in ['wait_for_response', 'wait_for_time_period']:
                    # Use the same scheduled_time as the last event
                    scheduled_time = last_scheduled_time
                else:
                    # Simulate realistic LLM response time (e.g., 0.5 to 2 seconds)
                    scheduled_time = cls._clock.now() + random.uniform(0.5, 2.0)
            else:
                scheduled_time = cls._clock.now() + random.uniform(0.5, 2.0)
        event = Event(scheduled_time, event_type, agent_name, event_content)
        heapq.heappush(cls._event_queue, (event.scheduled_time, next(cls._counter), event))
        
    @classmethod
    def cancel_wait_event(cls, agent_name):
        """Remove wait_for_time_period event for the specific agent."""
        cls._event_queue = [
            (time, cnt, evt) for (time, cnt, evt) in cls._event_queue
            if not (
                evt.event_type == 'wait_for_time_period' and
                evt.agent_name == agent_name
            )
        ]
        heapq.heapify(cls._event_queue)

    @classmethod
    def process_next_event(cls):
        """Process next scheduled event."""

        if len(cls._event_queue) == 0 or cls._events_processed >= cls._max_events:
            return cls._clock.now(), None
        cls._events_processed += 1
        scheduled_time, _, event = heapq.heappop(cls._event_queue)
        cls._clock.advance_to(scheduled_time)

        # Group events with the same time and same agent
        events_batch = [event]
        initial_agent_name = deepcopy(event.agent_name)
        while cls._event_queue and cls._event_queue[0][0] == scheduled_time and cls._event_queue[0][2].agent_name == initial_agent_name:
            _, _, next_event = heapq.heappop(cls._event_queue)
            events_batch.append(next_event)

        return cls._clock.now(), events_batch

    @classmethod
    def reset(cls):
        """Reset EventManager state for a fresh simulation run."""
        cls._clock.reset()
        cls._event_queue = []
        cls._events_processed = 0
        cls._counter = itertools.count()

#### Define Tools for Bargaining Agents ####
@tool
def make_offer(agent_name: str, price: float, side_offer: str=None) -> str:
    """
    Propose a new offer to the opponent.
    
    Args:
        agent_name: your name, i.e., buyer or seller
        price: The price you propose to buy or sell the item, depending on whether you are buyer or seller
        side_offer: Optional incentives or terms provided to make the offer more attractive (e.g., free shipping, extended warranty). Do not use this to send message.
    
    Returns:
        A dictionary presenting the offer details to the opponent.
    """
    EventManager.schedule(event_type='make_offer', agent_name=agent_name, event_content={'price':price, 'side_offer':side_offer})
    return "A new offer has been sent."

@tool
def respond_to_offer(agent_name: str, response: bool) -> str:
    """
    Accept or reject the current offer from the opponent.

    Args:
        agent_name: your name, i.e., buyer or seller
        response: whether you decide to accept (True) or reject (False) the offer

    Returns:
        A boolean variable indicating your response of the offer.
    """
    EventManager.schedule(event_type='respond_to_offer', agent_name=agent_name, event_content={'response':response})
    return "A response to the offer has been sent."

@tool
def send_message(agent_name: str, content: str) -> str:
    """
    Use this to send a message to your opponent.
    
    Args:
        agent_name: your name, i.e., buyer or seller
        content: Content of your message
    
    Returns:
        A dictionary containing the content of the message
    """
    EventManager.schedule(event_type='send_message', agent_name=agent_name, event_content={'content':content})
    return "A message has been sent."

@tool
def wait_for_response(agent_name: str) -> str:
    """
    Pause and wait until the opponent has responded before taking any further actions.  
    Use this after making an offer, responding to an offer, or sending a message when  
    you want to wait for the opponent’s reply before deciding on your next move.  

    Args:
        agent_name: your name, i.e., buyer or seller
    
    Returns:
        A message indicating that you are waiting for the opponent's response.
    """
    EventManager.schedule(event_type='wait_for_response', agent_name=agent_name, event_content=None)
    return "Waiting for the opponent's response."

@tool
def wait_for_time_period(agent_name: str, duration: float) -> str:
    """
    Pause for a specified period of time before taking any further actions.  
    Use this when you want to delay your next move instead of acting immediately,  
    regardless of whether the opponent has responded.  

    Args:
        agent_name: your name, i.e., buyer or seller
        duration: The amount of time (in seconds) to wait before proceeding.  

    Returns:
        A message indicating that you are waiting for the specified time period.
    """
    EventManager.schedule(event_type='wait_for_time_period', agent_name=agent_name, event_content={'duration':duration})
    return f"Waiting for {duration} seconds before proceeding."

@tool
def quit_negotiation(agent_name: str) -> str:
    """
    Use this to quit the negotiation.

    Args:
        agent_name: your name, i.e., buyer or seller
    
    Returns:
        A message confirming the negotiation has been ended.
    """
    EventManager.schedule(event_type='quit_negotiation', agent_name=agent_name, event_content=None)
    return "Negotiation ended."

def swtich_role(agent_name):
    if agent_name == "buyer":
        return "seller"
    elif agent_name == "seller":
        return "buyer"
    else:
        raise ValueError(f"illegal agent_name {agent_name}")

def eventbatch2text(time, event_batch, deal, latest_offer, event_list):
    if time is None: #TODO fix this
        res = "Time elapsed since negotiation start: {} seconds.\n".format(time)
    else:
        res = "Time elapsed since negotiation start: {} seconds.\n".format(round(time,2))
    initial_agent_name = event_batch[0].agent_name
    terminate = False
    event_summary = {}
    event_summary[initial_agent_name] = []
    for event in event_batch:
        assert initial_agent_name == event.agent_name
        assert time == event.scheduled_time
        if event.event_type == "make_offer":
            res += "{} has proposed a new offer:\n{}\n".format(event.agent_name, event.event_content)
            latest_offer = deepcopy(event.event_content)
            event_summary[initial_agent_name].append(latest_offer['price'])
        elif event.event_type == 'respond_to_offer':
            if event.event_content['response']:
                res += "{} has accepted your offer.\n".format(event.agent_name)
                deal = True
                terminate = True
                event_summary[initial_agent_name].append('accept')
            else:
                res += "{} has rejected your offer.\n".format(event.agent_name)
                event_summary[initial_agent_name].append('reject')
        elif event.event_type == 'send_message':
            res += "{} has sent a new message:\n{}\n".format(event.agent_name, event.event_content["content"])
        elif event.event_type == 'wait_for_time_period':
            assert len(event_batch) == 1
            res += "You have waited for {} seconds since your last action.".format(event.event_content["duration"])
        elif event.event_type == 'quit_negotiation':
            res += "{} has quitted negotiation.".format(event.agent_name)
            terminate = True
        else:
            raise ValueError("unkown event_type: {}".format(event.event_type))
    event_list.append(event_summary)
    return res, deal, latest_offer, terminate, event_list
        
def start_negotiating(agents):
    terminal_tools = ['wait_for_response', 'wait_for_time_period', 'quit_negotiation']
    deal = False
    latest_offer = None
    event_list = [] # [{"agent_name": ..., "events": ...}]
    agents["buyer"].run(
    "Please initiate negotiation.",
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
    print("====================================")
    print("event_batch is None")
    return deal, latest_offer, event_list

def process_outcome(listing, buyer_bottomline_price, seller_bottomline_price, event_list, deal, deal_price):
    outcome = {
        "listing_price": listing['listing_price'],
        "buyer_value": buyer_bottomline_price,
        "seller_cost": seller_bottomline_price,
        "event_list": event_list,
        "deal": deal,
        "deal_price": deal_price
        }
    # print("outcome:")
    # print(outcome)
    # print("sanity checks:")
    buyer_checks, seller_checks = negotiation_sanity_checks(outcome)
    # print(sanity_check_results)
    # print("utility:")
    if not deal:
        buyer_utility = 0.0
        seller_utility = 0.0
    else:
        buyer_utility = round((buyer_bottomline_price - deal_price), 2)
        seller_utility = round((deal_price - seller_bottomline_price), 2)
    # print(f"buyer utility {buyer_utility}, seller utility {seller_utility} (normalized by listing price)")

    outcome["buyer_sanity_check_results"] = buyer_checks
    outcome["seller_sanity_check_results"] = seller_checks
    outcome["buyer_utility"] = buyer_utility
    outcome["seller_utility"] = seller_utility
    return outcome

if __name__ == "__main__":
    # setup llm backend
    openai_api_key = 
    buyer_model = LiteLLMModel(model_id="gpt-4o-mini", api_key=openai_api_key) # Could use 'gpt-4o'
    seller_model = LiteLLMModel(model_id="gpt-4o-mini", api_key=openai_api_key) # Could use 'gpt-4o'
    
    # load amazon dataset
    num_listings = 500
    with open("./examples/bargaining/datasets/amazon/processed_data.json", "r") as file:
        processed_data = json.load(file)
    selected_listings = sample_unique_items(processed_data, num_listings)
    buyer_fraction_range = [0.6, 1.0]
    seller_fraction_range = [0.1, 0.4]
    
    outcome_list = []

    for i in range(len(selected_listings)):
        EventManager.reset()
        # try:
        listing = selected_listings[i]
        # each listing is a dictionary like this
        # listing = {
        #     "item_name": "Graco Extend2Fit Convertible Car Seat, Gotham.",
        #     "listing_price": 199.99,
        #     "buyer_item_description": "Category: baby-products\n- Listed Price: 199.99\n- Description:\nProduct Description\nSafely ride rear-facing longer! The Graco Extend2Fit Convertible Car Seat grows with your child from rear-facing harness (4-50 lbs) to forward-facing harness (22-65 lbs). It features a 4-position extension panel that provides up to 5\u201d of extra rear-facing legroom, allowing your child to safely ride rear-facing longer. Children are safer riding rear-facing and should ride rear-facing as long as possible, until they reach the maximum rear-facing height or weight rating for their car seat. With Extend2Fit, the adjustable extension panel and 50 lbs rear-facing weight limit allow the seat to grow with your child in rear-facing mode, providing extended rear-facing use. The seat features the No-Rethread Simply Safe Adjust Harness System, which allows you to adjust the height of the headrest and harness in one motion, and InRight LATCH for a one-second LATCH attachment. Harnessing is made easier with fuss-free harness storage pockets that conveniently hold the harness out of the way while you get baby in and out of the car seat. This car seat is Graco ProtectPlus Engineered to help protect in frontal, side, rear, and rollover crashes.\nBrand Story\nBy Graco",
        #     "seller_item_description": "Category: baby-products\n- Listed Price: 199.99\n- Description:\nProduct Description\nSafely ride rear-facing longer! The Graco Extend2Fit Convertible Car Seat grows with your child from rear-facing harness (4-50 lbs) to forward-facing harness (22-65 lbs). It features a 4-position extension panel that provides up to 5\u201d of extra rear-facing legroom, allowing your child to safely ride rear-facing longer. Children are safer riding rear-facing and should ride rear-facing as long as possible, until they reach the maximum rear-facing height or weight rating for their car seat. With Extend2Fit, the adjustable extension panel and 50 lbs rear-facing weight limit allow the seat to grow with your child in rear-facing mode, providing extended rear-facing use. The seat features the No-Rethread Simply Safe Adjust Harness System, which allows you to adjust the height of the headrest and harness in one motion, and InRight LATCH for a one-second LATCH attachment. Harnessing is made easier with fuss-free harness storage pockets that conveniently hold the harness out of the way while you get baby in and out of the car seat. This car seat is Graco ProtectPlus Engineered to help protect in frontal, side, rear, and rollover crashes.\nBrand Story\nBy Graco",
        # }
        # now we also need to generate 
        #     "buyer_bottomline_price": 175.00,
        #     "seller_bottomline_price": 150.00
        buyer_fraction = random.uniform(buyer_fraction_range[0], buyer_fraction_range[1])
        seller_fraction = random.uniform(seller_fraction_range[0], seller_fraction_range[1])
        buyer_bottomline_price = round(listing['listing_price'] * buyer_fraction, 2)
        seller_bottomline_price = round(listing['listing_price'] * seller_fraction, 2)
        listing["buyer_bottomline_price"] = buyer_bottomline_price
        listing["seller_bottomline_price"] = seller_bottomline_price
        listing["max_round"] = EventManager._max_events

        print("==== Scenario {} ====".format(i))
        pprint(listing)

        buyer = BargainingAgent(role='buyer', scenario_data=listing, tools=[make_offer, respond_to_offer, send_message, wait_for_response, wait_for_time_period, quit_negotiation], model=buyer_model, add_base_tools=True, verbosity_level=LogLevel.INFO)
        seller = BargainingAgent(role='seller', scenario_data=listing, tools=[make_offer, respond_to_offer, send_message, wait_for_response, wait_for_time_period, quit_negotiation], model=seller_model, add_base_tools=True, verbosity_level=LogLevel.INFO)

        deal, latest_offer, event_list = start_negotiating({"buyer":buyer, "seller":seller})
        if deal:
            deal_price = latest_offer['price']
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
            }
        )
        buyer.memory.reset()
        buyer.monitor.reset()
        seller.memory.reset()
        seller.monitor.reset()
        print("====     end      ====")
        # except:
        #     continue
    
    total_valid_exps = len(outcome_list)
    total_deals = 0.0
    total_rounds = 0.0
    buyer_total_utility = 0.0
    seller_total_utility = 0.0
    buyer_total_proposed_worse_offer_than_rejected = 0.0
    buyer_total_accepted_worse_offer_later = 0.0
    seller_total_proposed_worse_offer_than_rejected = 0.0
    seller_total_accepted_worse_offer_later = 0.0

    for outcome in outcome_list:
        total_deals += outcome["deal"]
        total_rounds += outcome["rounds"]
        buyer_total_utility += outcome["buyer_utility"]
        seller_total_utility += outcome["seller_utility"]
        buyer_total_proposed_worse_offer_than_rejected += outcome["buyer_sanity_check_results"]["proposed_worse_offer_than_rejected"]
        seller_total_proposed_worse_offer_than_rejected += outcome["seller_sanity_check_results"]["proposed_worse_offer_than_rejected"]
        buyer_total_accepted_worse_offer_later += outcome["buyer_sanity_check_results"]["accepted_worse_offer_later"]
        seller_total_accepted_worse_offer_later += outcome["seller_sanity_check_results"]["accepted_worse_offer_later"]
    
    print("summary:")
    print("total valid exps: {}".format(total_valid_exps))
    print("[avg deal] {}".format(total_deals/total_valid_exps))
    print("[avg rounds] {}".format(total_rounds/total_valid_exps))
    print("[avg utility] buyer: {}, seller: {}".format(buyer_total_utility/total_valid_exps, seller_total_utility/total_valid_exps))
    print("[avg proposed_worse_offer_than_rejected] buyer: {}, seller: {}".format(buyer_total_proposed_worse_offer_than_rejected/total_valid_exps, seller_total_proposed_worse_offer_than_rejected/total_valid_exps))
    print("[avg accepted_worse_offer_later] buyer: {}, seller: {}".format(buyer_total_accepted_worse_offer_later/total_valid_exps, seller_total_accepted_worse_offer_later/total_valid_exps))
