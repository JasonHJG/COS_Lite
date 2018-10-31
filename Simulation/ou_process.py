import numpy as np
import matplotlib.pyplot as plt


class Ornstein_Uhlenbeck:
    """
    define the OU process
    """

    def __init__(self, theta=np.log(2) / 5, mu=0, sigma=.15, p0=50, range = (0,100)):
        """
        simulate OU process according to:
        X_n+1 = X_n + theta(mu - X_n)dt + sigma sqrt(dt) N~(0,1)
        :param theta: float parameter
        :param mu: float parameter
        :param sigma: float parameter
        :param range: range for price movement
        """
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.p0 = p0
        self.range = range
        self.current_x = 0
        self.current_t = 0
        self.current_price = p0

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
        self.current_x = self.current_x + self.theta * (self.mu - self.current_x) * 1 + \
                      self.sigma * np.random.normal(0, 1)
        temp_price = round(np.exp(self.current_x) * self.p0, 1)
        if self.range[0] <= temp_price <= self.range[1]:
            self.current_price = temp_price
        self.current_t += 1

