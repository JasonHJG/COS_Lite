import numpy as np


class Generic_functions:
    """
    define functions for generic usage, i.e utility function, trading cost
    """

    @staticmethod
    def utility_function(dV, k=0.0001):
        """
        calculate the utility function according to utility = dV - .5k * dV^2
        :param dV: change in the price of each share
        :param k: float between 0 and 1, risk aversion parameter
        :return: utility
        """
        return dV - 0.5 * k * dV * dV

    @staticmethod
    def trading_cost(share, mul=10, ts=0.1):
        """
        calculate trading cost based on cost(share) = mul * ts *(abs(share)+0.01 * share^2)
        :param share: int, number of share
        :param mul: float, multiplier
        :param ts: float, ticker size
        :return: trading cost
        """
        return mul * ts * (np.abs(share) + 0.01 * share * share)
