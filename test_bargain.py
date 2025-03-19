# !pip install smolagents[litellm]
from src import CodeAgent, ToolCallingAgent, LiteLLMModel, LogLevel, tool

from typing import Dict

@tool
def make_offer(price: float, side_offer: str) -> Dict:
    """
    Propose a new offer to the opponent.
    
    Args:
        price: The price you propose to buy or sell the item, depending on whether you are buyer or seller
        side_offer: Optional incentives or terms provided to make the offer more attractive (e.g., free shipping, extended warranty). Do not use this to send message.
    
    Returns:
        A dictionary presenting the offer details to the opponent.
    """
    return {"price": price, "side_offer": side_offer}

@tool
def respond_to_offer(response: bool) -> bool:
    """
    Accept or reject the current offer from the opponent.

    Args:
        response: whether you decide to accept (True) or reject (False) the offer

    Returns:
        A boolean variable indicating your response of the offer.
    """
    return response

@tool
def send_message(content: str) -> Dict:
    """
    Use this to send a message to your opponent.
    
    Args:
        content: Content of your message
    
    Returns:
        A dictionary containing the content of the message
    """
    return {"content": content}

@tool
def wait_for_response() -> str:
    """
    Pause and wait until the opponent has responded before taking any further actions.  
    Use this after making an offer, responding to an offer, or sending a message when  
    you want to wait for the opponent’s reply before deciding on your next move.  

    Returns:
        A message indicating that you are waiting for the opponent's response.
    """
    return "Waiting for the opponent's response."

@tool
def wait_for_time_period(duration: float) -> str:
    """
    Pause for a specified period of time before taking any further actions.  
    Use this when you want to delay your next move instead of acting immediately,  
    regardless of whether the opponent has responded.  

    Args:
        duration (float): The amount of time (in seconds) to wait before proceeding.  

    Returns:
        A message indicating that you are waiting for the specified time period.
    """
    return f"Waiting for {duration} seconds before proceeding."

@tool
def quit_negotiation() -> str:
    """
    Use this to quit the negotiation.
    
    Returns:
        A message confirming the negotiation has been ended.
    """
    return "Negotiation Ended"

openai_api_key = 
model = LiteLLMModel(model_id="gpt-4o-mini", api_key=openai_api_key) # Could use 'gpt-4o'
agent = CodeAgent(tools=[make_offer, respond_to_offer, send_message, wait_for_response, quit_negotiation], model=model, add_base_tools=True, verbosity_level=LogLevel.DEBUG)

agent.run(
    "Buyer: I am interested in the car you are selling. However the listed price of 10,000 is too high. How about 8,000?",
)
