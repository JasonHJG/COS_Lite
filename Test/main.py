import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
from Simulation import Ornstein_Uhlenbeck
from Simulation import Random_mixture_process
from Simulation import Generic_functions
from Data import Trade_book
from Data import Player
from Model import Strategy
from Model import SLA
from Model import RWM
from Model import FTPL
import pandas as pd
import seaborn as sns
from scipy import stats


def main():
    np.random.seed(10)
    #ou = Random_mixture_process(p0=70, prob_list=[.25, .25, .25, .25])
    ou = Ornstein_Uhlenbeck()
    trade_cost = lambda x: Generic_functions.trading_cost(x, 10, 0.1)
    utility_func = lambda x: Generic_functions.utility_function(x, 0.0001)

    sla = SLA()
    rwm = RWM()
    ftpl = FTPL()
    strat_1 = Strategy(sla)
    strat_2 = Strategy(rwm)
    strat_3 = Strategy(ftpl)

    p1 = Player(price_process=ou, utility_function=utility_func, trading_cost=trade_cost, strategy=strat_3, model='ftpl')

    size = 1000
    for j in range(20):
        start = time.time()
        for i in range(size):
            p1.trade_greedy_one_step(.9 * 0.9 ** j)

        dates = list(p1.trade_book.book)[:-1]
        value_list = []
        for date in dates:
            value_list.append(p1.trade_book.book[date]['value'])
        value_array = np.array(value_list)
        initial_value = 1000000
        #values = np.cumsum(value_array) + initial_value
        returns = value_array / initial_value
        sharpe = np.mean(returns)/np.std(returns) * np.sqrt(252)
        print('sharpe ratio is:', sharpe)
        #print('weight is:', p1.strategy.learner.weight)
        p1.update_strategy(size, 1)
        end = time.time()
        print('iteration:',j+1,'time used is', end - start)


    # todo: generate many other price process and compute the sharpe
    p1.trade_book.clear()
    pnl = []
    initial_price = 60
    initial_time = 0
    p1.trade_book.add_state(initial_time, initial_price, 0)
    start = 0
    test_size = 1000
    while start != 200:
        print('iteration:', start+1)
        test_process = Random_mixture_process(prob_list=[.25, .25, .25, .25], p0 = initial_price, start_t= initial_time)
        p1.price_process = test_process
        for j in range(test_size):
            p1.trade_greedy_one_step(0.01)
        dates = list(p1.trade_book.book)[:-1]
        value_list = []
        for date in dates:
            value_list.append(p1.trade_book.book[date]['value'])
        value_array = np.array(value_list)
        initial_value = 1000000
        returns = value_array / initial_value
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        pnl.append(sharpe)
        p1.update_strategy(test_size, 1)
        start += 1
        initial_time, _ = p1.price_process.get_current_price()


    plt.figure()
    sns.distplot(pnl)
    plt.show()


    a = np.linspace(start=0, stop=100, num=41)
    b = []
    for i in a:
        b.append(p1.strategy.learner.predict((i, 500), [200, 100, 0, -100, -200]))

    plt.figure()
    plt.plot(a,b)
    plt.show(block = True)

    price = np.linspace(start=0,stop=100,num=21)
    q200 = []
    q100 = []
    q0 = []
    qn100 = []
    qn200 = []

    pos = 0
    for p in price:
        q200.append(p1.strategy.learner.qval((p, pos),[200]))
        q100.append(p1.strategy.learner.qval((p, pos),[100]))
        q0.append(p1.strategy.learner.qval((p, pos),[0]))
        qn100.append(p1.strategy.learner.qval((p, pos),[-100]))
        qn200.append(p1.strategy.learner.qval((p, pos),[-200]))
    plt.figure()
    plt.plot(price, q200, label = '200')
    plt.plot(price, q100, label = '100')
    plt.plot(price, q0, label = '0')
    plt.plot(price, qn100, label = '-100')
    plt.plot(price, qn200, label = '-200')
    plt.legend(loc=0)
    plt.show(block = True)

if __name__ == "__main__":
    main()
