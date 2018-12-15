import numpy as np


class Random_mixture_process:
    """
    random mixture of several price processes
    """
    def __init__(self, prob_list = [.25, .25, .25, .25], start_t = 0, p0 = 80, range = [1,100],
                 theta=np.log(2) / 5, mu=0, sigma=.15, r = 0.1, s = 0.05, threshold = 100):
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
        self.remaining_time = self.threshold
        self.indicator = np.random.choice([0, 1, 2, 3], p = self.prob_list)

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
        rand = np.random.random(4)
        self.prob_list = rand/np.sum(rand)

    def in_process(self):
        """
        check if one circle is finished
        :return: True if not finished, False if finished
        """
        if self.remaining_time > 1:
            return True
        else:
            return False

    def move_forward(self):
        """
        move the price one step forward
        """
        if self.in_process():
            self.remaining_time -= 1
        else:
            self.remaining_time = self.threshold
            self.remaining_time -= 1
            self.indicator = np.random.choice([0, 1, 2, 3], p=self.prob_list)

        indicator = self.indicator

        if indicator == 0: # OU process
            current_x = np.log(self.current_price / self.p0)
            current_x = current_x + self.theta * (self.mu - current_x) + self.sigma * np.random.normal(0, 1)
            temp_price = round(np.exp(current_x) * self.p0, 1)
            if self.range[0] <= temp_price <= self.range[1]:
                self.current_price = temp_price
            self.current_t += 1

        elif indicator == 1:
            current_x = np.log(self.current_price / (self.p0+10))
            current_x = current_x + self.theta * (self.mu - current_x) + self.sigma * np.random.normal(0, 1)
            temp_price = round(np.exp(current_x) * (self.p0-10), 1)
            if self.range[0] <= temp_price <= self.range[1]:
                self.current_price = temp_price
            self.current_t += 1

        elif indicator == 2:
            current_x = np.log(self.current_price / (self.p0-20))
            current_x = current_x + self.theta * (self.mu - current_x) + self.sigma * np.random.normal(0, 1)
            temp_price = round(np.exp(current_x) * (self.p0+10), 1)
            if self.range[0] <= temp_price <= self.range[1]:
                self.current_price = temp_price
            self.current_t += 1

        elif indicator == 3:
            current_x = np.log(self.current_price / (self.p0-10))
            current_x = current_x + self.theta * (self.mu - current_x) + self.sigma * np.random.normal(0, 1)
            temp_price = round(np.exp(current_x) * (self.p0 + 20), 1)
            if self.range[0] <= temp_price <= self.range[1]:
                self.current_price = temp_price
            self.current_t += 1

