import numpy as np


class Geometric_BM:
    """
    initialize one instance of geometric brownian motion
    """

    def __init__(self,current_t, cuurent_p, mu = 0.005 , sigma = 0.01, range = (0,100)):
        """
        initialize the a geometric brownian motion
        :param mu: return
        :param sigma: vol
        :param range: min and max for price
        :param current_t: current time
        :param cuurent_p: current price
        """
        self.mu = mu
        self.sigma = sigma
        self.range = range
        self.current_t = current_t
        self.current_p = cuurent_p

    def get_current_price(self):
        """
        get the current price
        :return: float price, int time
        """
        return self.current_t, self.current_price

    def move_forward(self):
        """
        move the price process one step forward
        """
        current_price = self.current_p
        next_price = current_price + self.mu * current_price + self.sigma * current_price * np.random.normal(0,1)
        self.current_t += 1
        self.current_p = next_price





