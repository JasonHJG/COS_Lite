import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from BackTester import utils
from IPython.display import display, HTML
from Data import Trade_book

class BackTester():
    """
    a back test platform that output performance metrics
    """
    def __init__(self, price_process, player):
        """
        initialize an instance of BackTester
        :param price_process: Price Process, a price process that mimics the behavior of a stock
        :param player: Player, a market player
        """
        self.price_process = price_process
        self.player = player
        self.df_trade = None
        self.trade_book = None

    def trade_one_step(self, trading_cost, utility_function, strategy, trade_book, threshold, action, epsilon=.001):
        """
        trade one step forward
        :param trading_cost:
        :param utility_function:
        :param strategy:
        :param trade_book:
        :param threshold:
        :param action:
        :param epsilon:
        :return:
        """
        self.trade_book = trade_book
        # get the player's state at t
        time_step, price, position = trade_book.get_recent_state()
        state = [price, position]
        # need to find the possible actions first
        possible_actions = []
        for action in action:
            if threshold[0] <= action + position <= threshold[1]:
                possible_actions.append(action)
        action = strategy.epsilon_greedy(state, possible_actions, epsilon)
        trade_book.add_action(time_step, action)
        self.price_process.move_forward()
        # update price and position at t+1
        next_time_step, next_price = self.price_process.get_current_price()
        next_position = position + action
        trade_book.add_state(next_time_step, next_price, next_position)
        # add player's value at t
        #dv = next_position * (next_price - price) - trading_cost(action)
        #utility = utility_function(dv)
        #trade_book.add_value(time_step, dv)
        #trade_book.add_utility(time_step, utility)


    def backtest(self, total_time=None, initial_value=1e5, threshold=None, action=None, ):
        """
        backtest for a certain time steps
        :param total_time: int, time steps
        :param initial_value: float, initial total cash for trading
        :param threshold: position threshold, if None get player.threshold
        :param action: action space, if None get player.action
        :return:
        """
        # if time == None, get price_process's total_time
        # prepare for real-time data price process
        if not total_time:
            try:
                total_time = self.price_process.total_time
            except AttributeError:
                pass

        if not threshold:
            threshold = self.player.threshold

        if not action:
            action = self.player.action

        # get the last state from training trade_book
        trade_book = Trade_book()
        trade_book.add_state(0, self.price_process.current_price, 0)
        # trade with epsilon 0
        for i in range(total_time):
            self.trade_one_step(self.player.trading_cost, self.player.utility_function, self.player.strategy,
                                trade_book, threshold, action, epsilon=.001)

        # convert the trade book to a dataframe
        book = trade_book.book
        df_trade = pd.DataFrame([book[i]['state'] for i in sorted(list(book))])

        # compute stock, cash and total value
        df_trade.loc[:, "stock"] = df_trade.loc[:, "price"] * df_trade.loc[:, "position"]
        df_trade.loc[:, "trading_cost"] = self.player.trading_cost(df_trade.position.diff())
        # note that action is made based on current price
        d_cash = (df_trade.price.shift() * df_trade.position.diff()).fillna(0) + df_trade.trading_cost
        df_trade.loc[:, "cash"] = initial_value - d_cash.cumsum()
        df_trade.loc[:, "value"] = df_trade.loc[:, "stock"] + df_trade.loc[:, "cash"]

        self.df_trade = df_trade

    def plot(self, figsize=(10,10)):
        """
        plot the performance
        :param figsize: tuple or list
        :return:
        """

        # if not backtested, backtest first
        if self.df_trade is None:
            self.backtest()

        # create fig
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        # plot value curve
        ax0 = plt.subplot(gs[0])
        ax0.set_title("Portfolio Value")
        self.df_trade.loc[:, ["stock", "cash", "value"]].plot(ax=ax0)

        # plot position bar
        ax1 = plt.subplot(gs[1])
        ax1.set_title("Position")
        plt.bar(self.df_trade.index, self.df_trade.position)

        plt.show()

    def print(self, html_print=False):
        """
        print performance stats
        :param html_print: bool, if True print in html format
        :return: df_summary, DataFrame
        """
        # if not backtested, backtest first
        if self.df_trade is None:
            self.backtest()

        # annualization factor
        af = 250

        # compute portfolio return
        rtn = self.df_trade.value.pct_change()

        # construct a summary data frame
        df_summary = pd.DataFrame(index=["Player"],
                                  columns=["annualized return", "annualized vol", "sharp ratio", 'hit rate', "turnover",
                                           "annualized cost rate", "max drawdown"])
        # compute metrics
        df_summary.loc["Player", "annualized return"] = rtn.mean()*af
        df_summary.loc["Player", "annualized vol"] = rtn.std() * np.sqrt(af)
        df_summary.loc["Player", "sharp ratio"] = df_summary.loc["Player", "annualized return"] / df_summary.loc["Player", "annualized vol"]
        df_summary.loc["Player", "turnover"] = utils.compute_turnover(position=self.df_trade.position)
        df_summary.loc["Player", "hit rate"] = utils.compute_hitrate(position=self.df_trade.position,
                                                                     price=self.df_trade.price)
        df_summary.loc["Player", "max drawdown"] = utils.compute_max_dd(value=self.df_trade.value)
        df_summary.loc["Player", "annualized cost rate"] = (self.df_trade.trading_cost / self.df_trade.value).mean()*af

        # display in html format
        if html_print:
            display(HTML(df_summary.to_html()))

        return df_summary


