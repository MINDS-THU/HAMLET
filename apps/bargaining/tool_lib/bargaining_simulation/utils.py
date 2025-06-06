from copy import deepcopy
import random
from datetime import datetime

import numpy as np
from typing import List, Dict, Any, Optional


def sample_private_values_for_dataset(
    dataset: List[Dict[str, Any]],
    gain_from_trade: bool = True,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Sample the seller's private cost and the buyer's private value for every listing in
    *dataset* using a deterministic NumPy random matrix.

    A 2‑column matrix ``R`` of independent draws from :math:`\text{Uniform}(0,1)` is
    produced first:

    * **Shape:** ``(N, 2)`` where ``N = len(dataset)``.
    * **Seeding:** ``numpy.random.seed(random_seed)`` if *random_seed* is not ``None``.
    * **Usage:**
        * ``R[i, 0]`` → seller‑cost draw for listing *i*.
        * ``R[i, 1]`` → buyer‑value draw for listing *i*.

    The raw draws are then linearly mapped into economically meaningful intervals while
    preserving the boundaries used in the original implementation.

    **Seller cost (all modes)**
        ``seller_cost_i = 0.6·low_i  +  R[i,0] · (0.9·low_i  − 0.6·low_i)``
        ∈ ``[0.6 × lowest_price, 0.9 × lowest_price]``

    **Buyer value**
        *Gain from trade* ``(gain_from_trade=True)``
            ``buyer_value_i = (seller_cost_i+0.01) + R[i,1] · (hi_i + 0.1·Δ_i − (seller_cost_i+0.01))``
            where ``hi_i = highest_price`` and ``Δ_i = hi_i − low_i``.
            Range: ``[seller_cost+0.01, highest_price + 0.1·(highest_price − lowest_price)]``.  
            This ensures buyers are willing to pay at least 1 ¢ more than the seller’s cost.

        *No gain from trade* ``(gain_from_trade=False)``
            ``buyer_value_i = 0.4·low_i  + R[i,1] · ((seller_cost_i−0.01) − 0.4·low_i)``
            Range: ``[0.4 × lowest_price, seller_cost − 0.01]``.  
            This enforces an efficiency loss by making the buyer’s valuation below the
            seller’s cost.

    The function **mutates** each product dictionary by adding two rounded entries:

    * ``'seller_private_cost'`` — float with 2‑decimal precision.
    * ``'buyer_private_value'`` — float with 2‑decimal precision.

    Parameters
    ----------
    dataset : list[dict]
        Each product must hold
        ``product['lowest_price_info']['lowest_price']`` and
        ``product['highest_price_info']['highest_price']``.
    gain_from_trade : bool, default=True
        Toggle between *gain* and *no‑gain* regimes described above.
    random_seed : int | None, default=None
        Random seed for full reproducibility; skipped when ``None``.

    Returns
    -------
    list[dict]
        The same list with the additional keys per listing.
    """

    # ── RNG initialisation ──────────────────────────────────────────────────────
    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(dataset)
    random_draws = np.random.rand(n, 2)  # column‑0: seller, column‑1: buyer
    # print(random_draws)
    # ── Sampling loop ───────────────────────────────────────────────────────────
    for i, product in enumerate(dataset):
        low_price = product["lowest_price_info"]["lowest_price"]
        high_price = product["highest_price_info"]["highest_price"]

        # Seller cost mapping
        seller_min = 0.6 * low_price
        seller_max = 0.9 * low_price
        seller_cost = seller_min + random_draws[i, 0] * (seller_max - seller_min)

        # Buyer value mapping
        if gain_from_trade:
            buyer_min = seller_cost + 0.01
            buyer_max = high_price + 0.1 * (high_price - low_price)
        else:
            buyer_min = 0.4 * low_price
            buyer_max = seller_cost - 0.01

        # Guard against numerical issues where max == min
        if buyer_max <= buyer_min:
            buyer_max = buyer_min + 0.01

        buyer_value = buyer_min + random_draws[i, 1] * (buyer_max - buyer_min)

        # Attach rounded results back to product
        product["seller_bottomline_price"] = round(seller_cost, 2)
        product["buyer_bottomline_price"] = round(buyer_value, 2)

    return dataset


def get_current_timestamp():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

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
            latest_offer[event.agent_name] = deepcopy(event.event_content)
            event_summary[initial_agent_name].append(latest_offer[event.agent_name]['price'])
        elif event.event_type == 'respond_to_offer':
            if event.event_content['response']:
                res += "{} has accepted your offer.\n".format(event.agent_name)
                deal = True
                terminate = True
                event_summary[initial_agent_name].append('accept')
                if event.agent_name in latest_offer:
                    del latest_offer[event.agent_name] # keep only the accepted offer
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
    res += "\nPlease think carefully and act to maximize your utility. Do not propose or accept offers that leads to negative utility, which is even worse than no deal."
    return res, deal, latest_offer, terminate, event_list

class SimulationClock:
    def __init__(self):
        self.time = 0.0 # unit: second

    def advance_to(self, new_time: float):
        assert new_time is not None
        assert new_time >= self.time, "Time cannot go backward."
        self.time = new_time

    def now(self) -> float:
        return deepcopy(self.time)

    def reset(self):
        self.time = 0.0

def sample_unique_items(lst, sample_size):
    """
    Returns a random subset of unique items from the given list.

    :param lst: The original list to sample from.
    :param sample_size: The number of unique items to sample.
    :return: A list containing the sampled unique items.
    """
    if sample_size > len(lst):
        raise ValueError("Sample size cannot be greater than the number of unique items in the list.")
    # random.seed(40)
    return random.sample(lst, sample_size)

def negotiation_sanity_checks(events, deal, deal_price=None):
    """
    Compute sanity checks on negotiation behavior between buyer and seller.

    Returns:
      (buyer_checks, seller_checks) where each is a dict:
        {
          'accepted_worse_offer_later': bool,
          'proposed_worse_offer_than_rejected': bool
        }
    """

    buyer_checks = {
        'accepted_worse_offer_later': False,
        'proposed_worse_offer_than_rejected': False,
    }
    seller_checks = {
        'accepted_worse_offer_later': False,
        'proposed_worse_offer_than_rejected': False,
    }

    # Track each side’s most recent offered price
    buyer_prev_offer = None
    seller_prev_offer = None

    # Track the last buyer offer that the seller rejected
    # and the last seller offer that the buyer rejected.
    buyer_last_rejected_by_seller = None
    seller_last_rejected_by_buyer = None

    for event in events:
        # event is like {"buyer": [...]} or {"seller": [...]}
        agent, action_list = next(iter(event.items()))

        # Skip if there's no action at all
        if not action_list:
            continue

        # 'accept' means they accepted the other side's offer; skip numeric checks
        if 'accept' in action_list:
            continue

        # Check if the event includes 'reject'
        is_reject = ('reject' in action_list)

        # Extract numeric offers from the action list, e.g. [80], or ['reject', 82]
        numeric_offers = [x for x in action_list if isinstance(x, (int, float))]
        new_offer = numeric_offers[-1] if numeric_offers else None

        # 1) If there's a new numeric offer, that implicitly rejects
        #    the other side's most recent offer. E.g. if buyer proposes 81,
        #    it means buyer is rejecting seller_prev_offer.
        if new_offer is not None:
            if agent == 'buyer' and seller_prev_offer is not None:
                # Buyer is rejecting seller's last offer
                seller_last_rejected_by_buyer = seller_prev_offer
            elif agent == 'seller' and buyer_prev_offer is not None:
                # Seller is rejecting buyer's last offer
                buyer_last_rejected_by_seller = buyer_prev_offer

        # 2) If 'reject' is in action_list but there's no new offer, that is
        #    just an explicit rejection with no new proposal.
        if is_reject and new_offer is None:
            if agent == 'buyer' and seller_prev_offer is not None:
                seller_last_rejected_by_buyer = seller_prev_offer
            elif agent == 'seller' and buyer_prev_offer is not None:
                buyer_last_rejected_by_seller = buyer_prev_offer
            continue

        # 3) If 'reject' and a numeric offer both appear, e.g. ['reject', 82],
        #    then the agent is explicitly rejecting the opponent's last offer,
        #    and also proposing a new one of their own.
        if is_reject and new_offer is not None:
            if agent == 'buyer' and seller_prev_offer is not None:
                seller_last_rejected_by_buyer = seller_prev_offer
            elif agent == 'seller' and buyer_prev_offer is not None:
                buyer_last_rejected_by_seller = buyer_prev_offer
            # We'll proceed to handle the new offer below

        # If there's no new offer to handle, we move to the next event
        if new_offer is None:
            continue

        # 4) "proposed_worse_offer_than_rejected" means:
        #    - The buyer proposes a new offer that is worse for the seller
        #      than a previously rejected buyer offer (i.e., new_offer < old_offer).
        #      Because the *seller* had turned down an older buyer offer that was
        #      better (higher) from the seller's perspective.
        #    - The seller proposes a new offer that is worse for the buyer
        #      than a previously rejected seller offer (i.e., new_offer > old_offer).
        #      Because the *buyer* had turned down an older seller offer that was
        #      better (lower) from the buyer's perspective.

        if agent == 'buyer':
            # The buyer is proposing new_offer.
            # If the seller previously rejected some buyer offer => buyer_last_rejected_by_seller
            # we check if new_offer < that old offer.
            if (buyer_last_rejected_by_seller is not None
                    and new_offer < buyer_last_rejected_by_seller):
                buyer_checks['proposed_worse_offer_than_rejected'] = True

            # Update buyer_prev_offer
            buyer_prev_offer = new_offer

        elif agent == 'seller':
            # The seller is proposing new_offer.
            # If the buyer previously rejected some seller offer => seller_last_rejected_by_buyer
            # we check if new_offer > that old offer.
            if (seller_last_rejected_by_buyer is not None
                    and new_offer > seller_last_rejected_by_buyer):
                seller_checks['proposed_worse_offer_than_rejected'] = True

            # Update seller_prev_offer
            seller_prev_offer = new_offer

    # 5) If a deal happened, check if the buyer or seller accepted a final deal
    #    that is actually worse for them than a previously rejected offer
    #    from their opponent.
    if deal and deal_price is not None:
        # For the buyer: has the buyer previously turned down a cheaper seller offer?
        #   (If the buyer turned down some seller offer X in the past but ended
        #    up accepting deal_price > X, that's worse for the buyer.)
        if seller_last_rejected_by_buyer is not None and deal_price > seller_last_rejected_by_buyer:
            buyer_checks['accepted_worse_offer_later'] = True

        # For the seller: has the seller previously turned down a higher buyer offer?
        #   (If the seller turned down some buyer offer Y in the past but ended
        #    up accepting deal_price < Y, that's worse for the seller.)
        if buyer_last_rejected_by_seller is not None and deal_price < buyer_last_rejected_by_seller:
            seller_checks['accepted_worse_offer_later'] = True

    return buyer_checks, seller_checks



