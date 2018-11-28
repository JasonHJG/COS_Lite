import numpy as np


class Random_mixture_process:
    """
    random mixture of several price processes
    """
    def __init__(self, prob_list = [0.8, 0.2], start_t = 0, p0 = 50, range = [0,100],
                 theta=np.log(2) / 5, mu=0, sigma=.15, r = 0.05/ 252, s = 0.2 / 17, threshold = 10000):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.r = r
        self.s = s
        self.p0 = p0
        self.prob_list = prob_list
        self.current_t = start_t
        self.current_price = p0
        self.range = range
        self.threshold = threshold

    def get_current_price(self):
        """
        get the current price
        :return: float price, int time
        """
        return self.current_t, self.current_price

    def change(self):
        """
        change the market regime once time reaches threshold
        """
        rand = np.random.random()
        if 0<self.prob_list[0] + rand<1:
            self.prob_list[0] = self.prob_list[0] + rand
            self.prob_list[1] = 1 - self.prob_list[0]

    def move_forward(self):
        """
        move the price one step forward
        """
        if self.current_t%self.threshold == 0:
            self.change()

        indicator = np.random.choice([0,1], p = self.prob_list)
        if indicator == 0: # OU process
            current_x = np.log(self.current_price / self.p0)
            current_x = current_x + self.theta * (self.mu - current_x) + self.sigma * np.random.normal(0, 1)
            temp_price = round(np.exp(current_x) * self.p0, 1)
            if self.range[0] <= temp_price <= self.range[1]:
                self.current_price = temp_price
            self.current_t += 1

        else: #GBM
            current_price = self.current_price
            next_price = current_price + self.r * current_price + self.s * current_price * np.random.normal(0, 1)
            self.current_t += 1
            self.current_price = next_price
