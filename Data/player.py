import numpy as np
import time
from .trade_book import Trade_book


class Player:
    """
    define the player, who is able to buy and sell a stock he is holding
    the player has his own strategy, utility function
    the player needs to record each step the state and action
    """

    def __init__(self, price_process, utility_function, trading_cost, strategy, gamma = 0.999, action=[-200, -100, 0, 100, 200],
                 threshold=(-1000, 1000), model='rwm'):
        """
        initialize an instance of player in the stock market
        :param price_process: a price process that mimics the behavior of a stock
        :param utility_function: the utility function of the player
        :param trading_cost: the trading cost function
        :param strategy: the RL strategy of the player
        :param gamma: [0.8], float, speed of diminishing utility
        :param action: list of possible movement of position
        """
        self.price_process = price_process
        self.utility_function = utility_function
        self.trading_cost = trading_cost
        self.strategy = strategy
        self.action = action
        self.gamma = gamma
        time, price = price_process.get_current_price()
        self.trade_book = Trade_book()
        self.trade_book.add_state(time, price, 0) #assume no initial position
        self.utility = {}
        self.threshold = threshold
        self.model = model

    def progress(self):
        """
        the stock the player is observing moves one step forward
        """
        self.price_process.move_forward()

    def trade_greedy_one_step(self, epsilon=.05):
        """
        trade one step forward
        """
        # get the player's state at t
        time_step, price, position = self.trade_book.get_recent_state()
        state = [price, position]
        # need to find the possible actions first
        possible_actions = []
        for action in self.action:
            if self.threshold[0] <= action + position <= self.threshold[1]:
                possible_actions.append(action)
        action = self.strategy.epsilon_greedy(state, possible_actions, epsilon)
        self.trade_book.add_action(time_step, action)
        self.progress()
        # update price and position at t+1
        next_time_step, next_price = self.price_process.get_current_price()
        next_position = position + action
        self.trade_book.add_state(next_time_step, next_price, next_position)
        # add player's utility at t
        dv = next_position * (next_price - price) - self.trading_cost(action)
        utility = self.utility_function(dv)
        self.trade_book.add_utility(time_step, utility)
        if self.model == 'rwm':
            best_action = self.feedback_best_action(possible_actions, next_price- price, position)
            self.strategy.learner.adjust_weight(best_action)

    def feedback_best_action(self, possible_actions, delta_price, postion):
        """
        decide which action is the best action
        :param possible_actions: list of possible action
        :param delta_price: change in price
        :param postion: previous position
        :return: int best action
        """
        utility = np.zeros(len(possible_actions))
        for i in range(len(possible_actions)):
            dv = (postion + possible_actions[i]) * delta_price - self.trading_cost(possible_actions[i])
            utility[i] = self.utility_function(dv)
        return possible_actions[np.argmax(utility)]

    def update_strategy(self, look_back=50000, length_of_state=1):
        """
        update Q function in strategy based on past observations
        :param look_back: int, number of steps to look back
        :param length_of_state: [1], number of steps to take as one state
        """
        trade_book = self.trade_book.book
        self.strategy.upgrade(trade_book, self.gamma)
        self._clean_trade_book()

    def _clean_trade_book(self):
        """
        clean trade book so that it only contains the most recent state
        """
        time_step, price, position = self.trade_book.get_recent_state()
        self.trade_book.clear()
        self.trade_book.add_state(time_step, price, position)
