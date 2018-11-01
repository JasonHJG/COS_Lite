import numpy as np


class Strategy:
    """
    exploit and explore strategy
    """
    def __init__(self, learner):
        """
        initialize an instance of strategy class
        it is able to take observations of state-action pairs and use the learner inside it to train and predict
        :param learner: some supervised learner
        """
        self.learner = learner

    def epsilon_greedy(self, observed_states, actions, epsilon=0.05):
        """
        trade according to epsilon greedy algorithm:
        with probability epsilon to trade randomly,
        with probability 1-epsilon not to trade (action = 0)
        :param observed_states: state as a list of variables
        :param actions: int list of actions allowed to take
        :param epsilon: float probability to trade randomly
        :return: int action to make
        """
        # with probability epsilon, trading randomly
        ran = np.random.random()
        if ran < epsilon:  # trade randomly
            rand_index = np.random.randint(len(actions))
            action = actions[rand_index]
        # with probability 1-epsilon, trading greedily
        else:
            action = self.learner.predict(observed_states, actions)
        return action

    def upgrade(self, trade_book, gamma):
        """
        upgrade the learner with observations of one-step Sarsa target
        notice trade book at time t matches the utility at time t in our setting,
        :param trade_book: dictionary of time_step(t):[price(t), position(t), cash(t), action(t)]
        :param gamma: float between (0,1) discounting factor for that person
        """
        time_steps = list(trade_book)
        time_steps.sort()
        X = []
        y = []
        print(len(time_steps))
        for i in range(len(time_steps)-2):
            price = trade_book[time_steps[i]]['state']['price']
            position = trade_book[time_steps[i]]['state']['position']
            state = np.array((price, position))
            action = trade_book[time_steps[i]]['action']
            X.append(np.r_[state, action])
            next_price = trade_book[time_steps[i+1]]['state']['price']
            next_position = trade_book[time_steps[i+1]]['state']['position']
            next_state = np.array((next_price, next_position))
            next_action = trade_book[time_steps[i+1]]['action']
            next_utility = trade_book[time_steps[i]]['utility']
            y.append(next_utility + gamma * self.learner.qval(next_state, next_action))
        self.learner.fit(X, y)

