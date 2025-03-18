def negotiation_sanity_checks(negotiation_data):
    """
    Compute sanity checks on negotiation behavior between buyer and seller.

    Parameters:
    negotiation_data (dict): Dictionary containing the negotiation events and outcomes.

    Returns:
    tuple: Two dictionaries summarizing the sanity check results for buyer and seller.
    """

    events = negotiation_data['event_list']
    deal = negotiation_data['deal']
    deal_price = negotiation_data.get('deal_price')

    buyer_checks = {
        'accepted_worse_offer_later': False,
        'proposed_worse_offer_than_rejected': False,
    }

    seller_checks = {
        'accepted_worse_offer_later': False,
        'proposed_worse_offer_than_rejected': False,
    }

    buyer_prev_offer = None
    seller_prev_offer = None

    buyer_rejected_offers = []
    seller_rejected_offers = []

    buyer_last_rejected_by_seller = None
    seller_last_rejected_by_buyer = None

    for event in negotiation_data['event_list']:
        agent, action = next(iter(event.items()))

        # Check if agent explicitly rejects without proposing new offer
        if len(action) == 0 or (len(action) == 1 and action[0] == 'reject'):
            # Explicit rejection without new offer
            if agent == 'buyer' and seller_prev_offer is not None:
                buyer_rejected_offer = seller_prev_offer
                seller_last_rejected_by_buyer = seller_prev_offer
                buyer_prev_offer = None
            elif agent == 'seller' and buyer_prev_offer is not None:
                seller_last_rejected_by_buyer = buyer_prev_offer
            continue

        if action[0] == 'accept':
            continue

        offer = action[0]

        if agent == 'buyer':
            # Proposing new offer rejects seller's previous offer
            if seller_prev_offer is not None:
                seller_last_rejected_by_buyer = seller_prev_offer

            # Check if buyer proposed worse (lower) offer for seller after rejection
            if buyer_last_rejected_by_seller is not None and offer < buyer_last_rejected_by_seller:
                buyer_checks['proposed_worse_offer_than_rejected'] = True

            buyer_prev_offer = offer

        elif agent == 'seller':
            # Proposing new offer rejects buyer's previous offer
            if buyer_prev_offer is not None:
                buyer_last_rejected_by_seller = buyer_prev_offer

            # Check if seller proposes worse offer for buyer after rejection
            if seller_last_rejected_by_buyer is not None and offer > seller_last_rejected_by_buyer:
                seller_checks['proposed_worse_offer_than_rejected'] = True

            seller_prev_offer = offer

        # Update previous offers
        if agent == 'buyer':
            buyer_prev_offer = offer
        elif agent == 'seller':
            seller_prev_offer = offer

    # Check if an agent accepted a worse offer later
    if negotiation_data['deal']:
        if buyer_last_rejected_by_seller is not None and deal_price > buyer_last_rejected_by_seller:
            buyer_checks['accepted_worse_offer_later'] = True
        if seller_last_rejected_by_buyer is not None and deal_price < seller_last_rejected_by_buyer:
            seller_checks['accepted_worse_offer_later'] = True

    return buyer_checks, seller_checks


# Example Usage:
negotiation_data = {
    'listing_price': 69.99,
    'buyer_value': 69.23554276115871,
    'seller_cost': 21.938333429894893,
    'event_list': [
        {'buyer': [69.0]},
        {'seller': [69.5]},
        {'buyer': [69.2, 'reject']},
        {'seller': [70]},
        {'buyer': ['accept']},
    ],
    'deal': True,
    'deal_price': 70
}

# Compute sanity checks
buyer_result, seller_result = negotiation_sanity_checks(negotiation_data)

print("Buyer Checks:", buyer_result)
print("Seller Checks:", seller_result)
