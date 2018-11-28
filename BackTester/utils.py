import pandas as pd
import numpy as np


def compute_hitrate(position, price):
    """
    Compute the hit rate (#{action hit price change} / #{all})
    :param position: Series
    :param price: Series
    :return: hitrate
    """
    action = position.diff()
    price_change = price.diff()

    # product
    prod = action.fillna(0) * price_change.fillna(0)

    # compute all bet
    prod = prod[position.fillna(0).abs() >= 1e-6]
    all_bet = prod.count()

    # compute positive bet
    prod = prod[prod > 1e-6]
    pos_bet = prod.count()

    return pos_bet / all_bet


def compute_max_dd(value):
    """
    compute max draw down
    :param value: a series or data frame
    :return: max draw down
    """
    value = value.ffill()
    cum_max = np.maximum.accumulate(value.ffill())
    dd = 1. - value / cum_max
    mdd = dd.max()
    return mdd if mdd < 1. else 1.


def compute_turnover(position):
    """
    compute turnover
    :param position:
    :return: turnover in shares
    """
    thruput = abs(position - position.shift(1)).sum()
    return thruput / 2.
