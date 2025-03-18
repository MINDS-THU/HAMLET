from copy import deepcopy
import random

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

def negotiation_sanity_checks(negotiation_data):
    """
    Compute sanity checks on negotiation behavior between buyer and seller.

    negotiation_data is a dict like:
    {
        "listing_price": float,
        "buyer_value": float,
        "seller_cost": float,
        "event_list": [
           {"buyer": [80]},
           {"seller": ["reject", 82]},
           {"buyer": [81]},
           {"seller": ["accept"]}
        ],
        "deal": bool,
        "deal_price": float or None
    }

    Returns:
      (buyer_checks, seller_checks) where each is a dict:
        {
          'accepted_worse_offer_later': bool,
          'proposed_worse_offer_than_rejected': bool
        }
    """

    events = negotiation_data['event_list']
    deal = negotiation_data['deal']
    deal_price = negotiation_data.get('deal_price', None)

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



