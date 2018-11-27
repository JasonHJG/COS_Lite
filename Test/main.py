import sys
import numpy as np
import matplotlib.pyplot as plt
import time

from Simulation import Ornstein_Uhlenbeck
from Simulation import Generic_functions
from Data import Trade_book
from Data import Player
from Model import Strategy
from Model import SLA
from Model import RWM

def main():
    ou = Ornstein_Uhlenbeck()
    trade_cost = lambda x: Generic_functions.trading_cost(x, 10, 0.1)
    utility_func = lambda x: Generic_functions.utility_function(x, 0.0001)

    sla = SLA()
    rwm = RWM()
    strat = Strategy(rwm)

    p1 = Player(price_process= ou, utility_function=utility_func, trading_cost=trade_cost, strategy=strat, model='rwm')

    size = 50000
    for j in range(15):
        start = time.time()
        for i in range(size):
            p1.trade_greedy_one_step(.5 * 0.9 ** j)
        print(p1.strategy.learner.weight)
        p1.update_strategy(size, 1)
        end = time.time()
        print('iteration:',j+1,'time used is', end - start)

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