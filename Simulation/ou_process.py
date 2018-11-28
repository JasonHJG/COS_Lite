import numpy as np
import matplotlib.pyplot as plt


class Ornstein_Uhlenbeck:
    """
    define the OU process
    """

    def __init__(self, current_t = 0, cuurent_p = 50, theta=np.log(2) / 5, mu=0, sigma=.15, range = (0,100)):
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
        self.p0 = cuurent_p
        self.range = range
        self.current_t = current_t
        self.current_price = self.p0

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
        current_x = np.log(self.current_price / self.p0)
        current_x = current_x + self.theta * (self.mu - current_x) + self.sigma * np.random.normal(0, 1)
        temp_price = round(np.exp(current_x) * self.p0, 1)
        if self.range[0] <= temp_price <= self.range[1]:
            self.current_price = temp_price
        self.current_t += 1

