import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from BackTester import utils
from IPython.display import display, HTML


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

    def backtest(self, trading_cost, total_time=None, initial_value=1e5, ):
        """
        backtest for a certain time steps
        :param trading_cost: function
        :param total_time: int, time steps
        :param initial_value: float, initial total cash for trading
        :return:
        """
        # if time == None, get price_process's total_time
        # prepare for real-time data price process
        if not total_time:
            try:
                total_time = self.price_process.total_time
            except AttributeError:
                pass

        # replace the player's observable process with a new price_process
        self.player.price_process = self.price_process

        # trade with epsilon .01
        for i in range(total_time):
            self.player.trade_greedy_one_step(.01)

        # convert the trade book to a dataframe
        trade_book = self.player.trade_book.book
        df_trade = pd.DataFrame([trade_book[i]['state'] for i in sorted(list(trade_book))])

        # compute stock, cash and total value
        df_trade.loc[:, "stock"] = df_trade.loc[:, "price"] * df_trade.loc[:, "position"]
        df_trade.loc[:, "trading_cost"] = trading_cost(df_trade.position.diff())
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


