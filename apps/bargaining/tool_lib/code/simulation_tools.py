import heapq, itertools
from collections import namedtuple
from .utils import SimulationClock
import random
from copy import deepcopy
from smolagents import tool, Tool

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

class SearchPrice(Tool):
    name = "search_price"
    description = "search for the price history of the item."
    inputs = {}
    output_type = "string"

    def __init__(self, highest_price_info, lowest_price_info):
        super().__init__()
        self.highest_price_info = highest_price_info
        self.lowest_price_info = lowest_price_info

    def forward(self) -> str:
        res = "Highest history price of the item:"
        res += f"{self.highest_price_info['highest_price']} on {self.highest_price_info['highest_price_date']}\n"
        res += "Lowest history price of the item:"
        res += f"{self.lowest_price_info['lowest_price']} on {self.lowest_price_info['lowest_price_date']}"
        return res