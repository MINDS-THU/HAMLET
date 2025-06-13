from copy import deepcopy
import random
from datetime import datetime

import numpy as np
from typing import List, Dict, Any, Optional

from typing import List, Dict, Any, Optional
import numpy as np


def sample_private_values_for_dataset(
    dataset: List[Dict[str, Any]],
    random_seed: Optional[int] = None,
    mode: str = "uniform",
    *,
    gamma: float = 0.10,
    low_factor: float = 0.40,
    seller_low_mult: float = 0.60,
    seller_high_mult: float = 0.90,
    buffer: float = 0.01,
) -> List[Dict[str, Any]]:
    """Generate *private* seller costs and buyer values for every listing.

    This implementation follows **Option ② — independent draws with
    post‑stratification**:

    1.  **Seller cost** is always sampled from a fixed band strictly below the
        historical *lowest* market price::

            c_s ~ U[ seller_low_mult × L , seller_high_mult × L ]

        with the defaults 0.60 × L and 0.90 × L.

    2.  **Buyer value** is first drawn *independently* of *c_s* from a wide,
        realistic interval that spans low‑ball offers to exuberant willingness
        to pay::

            v_b ~ U[ low_factor × L , H + γ (H − L) ]

        where *L* = ``lowest_price`` and *H* = ``highest_price``.

    3.  A *regime* is enforced **only if** ``mode`` is set to
       ``"force_gain_from_trade"`` or ``"force_no_gain_from_trade"``.
       This is done via an *acceptance–rejection* loop that redraws **only the
       buyer** until the inequality holds::

            force_gain_from_trade   →   v_b  >  c_s  + buffer
            force_no_gain_from_trade→   v_b  <  c_s  − buffer

    4.  A deterministic RNG (`numpy.random.RandomState`) guarantees **full
       reproducibility**: the same ``random_seed`` yields the same sequence of
       draws and therefore the *same* (cost, value) pairs, while still allowing
       variation across listings.

    Parameters
    ----------
    dataset : list[dict]
        Each listing dictionary must expose::

            product['lowest_price_info']['lowest_price']
            product['highest_price_info']['highest_price']

    random_seed : int | None, default=None
        Seed for the internal RNG.  ``None`` → non‑deterministic.

    mode : {'uniform', 'force_gain_from_trade', 'force_no_gain_from_trade'}
        * ``'uniform'`` ‑ sample buyer values independently; *ex post* share of
          gain‑from‑trade episodes is whatever arises naturally.
        * ``'force_gain_from_trade'`` ‑ acceptance–rejection to guarantee
          *v_b > c_s*.
        * ``'force_no_gain_from_trade'`` ‑ acceptance–rejection to guarantee
          *v_b < c_s*.

    gamma : float, default=0.10
        Head‑room fraction above the historical high when sampling buyer values.
        Buyer upper bound = *H + γ (H − L)*.

    low_factor : float, default=0.40
        Multiplier for the buyer lower bound (× L).

    seller_low_mult, seller_high_mult : float, defaults=(0.60, 0.90)
        Multipliers for the seller cost band (× L).

    buffer : float, default=0.01
        Safety gap (in *currency units*) used in the inequality checks to avoid
        equality edge‑cases.

    Returns
    -------
    list[dict]
        The *same* list with two extra key–value pairs added to each listing::

            'seller_bottomline_price'  → rounded seller cost
            'buyer_bottomline_price'   → rounded buyer value

    Notes
    -----
    • The function **mutates** the original dataset in place and also returns it
      for convenience.
    • Acceptance–rejection loops are expected to finish in ≈ 2 draws on average
      when *buffer* is small and the buyer range is wide.
    """

    # ────────────────────────────── RNG setup ────────────────────────────────
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random

    # ───────────────────────── Main sampling loop ────────────────────────────
    for product in dataset:
        # --- Extract price anchors ------------------------------------------------
        L = product["lowest_price_info"]["lowest_price"]
        H = product["highest_price_info"]["highest_price"]
        Δ = H - L  # price range width

        # --- Seller cost: one draw, never resampled ------------------------------
        seller_min = seller_low_mult * L
        seller_max = seller_high_mult * L
        seller_cost = seller_min + rng.rand() * (seller_max - seller_min)

        # --- Buyer value: depends on *mode* --------------------------------------
        buyer_min = low_factor * L
        buyer_max = H + gamma * Δ

        def draw_buyer() -> float:
            """Draw a candidate buyer value from the *base* uniform."""
            return buyer_min + rng.rand() * (buyer_max - buyer_min)

        buyer_value = draw_buyer()

        if mode == "force_gain_from_trade":
            # Keep drawing until buyer is willing to pay *strictly* above cost.
            while buyer_value <= seller_cost + buffer:
                buyer_value = draw_buyer()

        elif mode == "force_no_gain_from_trade":
            # Keep drawing until buyer value is *strictly* below cost.
            while buyer_value >= seller_cost - buffer:
                buyer_value = draw_buyer()

        elif mode != "uniform":
            raise ValueError(
                "mode must be one of {'uniform', 'force_gain_from_trade', 'force_no_gain_from_trade'}"
            )

        # --- Attach rounded results back to product ------------------------------
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



