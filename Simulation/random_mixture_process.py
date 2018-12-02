import numpy as np


class Random_mixture_process:
    """
    random mixture of several price processes
    """
    def __init__(self, prob_list = [.5, .2, .3], start_t = 0, p0 = 50, range = [5,100],
                 theta=np.log(2) / 5, mu=0, sigma=.15, r = 0.1, s = 0.1, threshold = 1000):
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
        rand = np.random.normal(0, 0.1, 3)
        new_prob = self.prob_list + (rand-np.sum(rand)/3)
        if np.min(new_prob)>0:
            self.prob_list = new_prob

    def move_forward(self):
        """
        move the price one step forward
        """
        if self.current_t%self.threshold == 0:
            self.change()

        indicator = np.random.choice([0,1,2], p = self.prob_list)
        if indicator == 0: # OU process
            current_x = np.log(self.current_price / self.p0)
            current_x = current_x + self.theta * (self.mu - current_x) + self.sigma * np.random.normal(0, 1)
            temp_price = round(np.exp(current_x) * self.p0, 1)
            if self.range[0] <= temp_price <= self.range[1]:
                self.current_price = temp_price
            self.current_t += 1

        elif indicator == 1 : #GBM + drift
            current_price = self.current_price
            next_price = current_price + self.r * current_price + self.s * current_price * np.random.normal(0, 1)
            if self.range[0] <= next_price <= self.range[1]:
                self.current_price = next_price
            self.current_t += 1

        elif indicator == 2:  # GBM - drift
            current_price = self.current_price
            next_price = current_price - self.r * current_price + self.s * current_price * np.random.normal(0, 1)
            if self.range[0] <= next_price <= self.range[1]:
                self.current_price = next_price
            self.current_t += 1
