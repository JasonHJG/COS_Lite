from collections import defaultdict


class Trade_book:
    """
    keep a record of all the transactions and the net worth
    Trade_book has date:{'state':{'price':float, 'position':float}, 'action':int, 'utility':float}
    """
    def __init__(self):
        """
        initialize an instance of trade book
        """
        self.book = defaultdict(dict)
        self.recent_time = None

    def get_recent_state(self):
        """
        get the most recent price
        :return: time_step, price, position
        """
        recent_time_step = self.recent_time
        price = self.book[recent_time_step]['state']['price']
        position = self.book[recent_time_step]['state']['position']
        return recent_time_step, price, position

    def add_state(self, time_step, price, position):
        """
        record the state to the trade book
        :param time_step: index for time step
        :param price: float price for each share
        :param position: int position for the share
        """
        self.book[time_step]={'state':{'price':price, 'position':position}}
        self.recent_time = time_step

    def add_action(self, time_step, action):
        """
        record action to the trade book
        :param time_step: index for time step
        :param action: int, change in the position
        """
        self.book[time_step]['action'] = action

    def add_utility(self, time_step, utility):
        """
        record the utility to the trade book
        :param time_step: index for time step
        :param utility: float utility
        """
        self.book[time_step]['utility'] = utility

    def clear(self):
        """
        clear the trade book, set it to None
        """
        self.book.clear()
