from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import random


class COS:
    """
    Continuous state Online Supervised reinforcement learning
    """

    def __init__(self, sk_regressor=DecisionTreeRegressor()):
        """
        initialize an instance of COS learner
        :param sk_regressor:[DecisionTreeRegressor], type of learner for supervised learning
        """
        self.regressor = sk_regressor
        self.supervised_learners = []
        self.learner_weights = None
        self.recent_guess = None

    def adjust_score(self, best_action):
        """
        adjust the score according to best action
        :param best_action: int, action that optimiza the utility
        """
        if self.learner_weights is None:
            return
        for i in range(len(self.recent_guess)):
            if self.recent_guess[i] != best_action:
                self.learner_weights[i] += 1

    def update_recent_guess(self, state, possible_actions):
        """
        update recent guess from the state and possible actions
        :param state: list of price and position
        :param possible_actions: int list of possible actions
        """
        if not self.supervised_learners:
            return
        # for each regressor
        for i in range(len(self.recent_guess)):
            state_action_value = np.zeros(len(possible_actions))
            # for all the possible actions
            for j in range(len(possible_actions)):
                x = np.r_[state, possible_actions[j]].reshape((1, -1))
                state_action_value[j] = self.supervised_learners[i].predict(x)
            self.recent_guess[i] = possible_actions[np.argmax(state_action_value)]

    def ftpl(self, scale=1):
        """
        give the best possible action based on the follow the best perturbed leader
        :param scale: parameter for exponential distribution
        :param state: state, list of price and position
        :param possible_actions: possible actions to take
        :return: int action, int best regressor id
        """
        if not self.supervised_learners:
            return 0, -1
        length = len(self.learner_weights)
        disturbed_weight = np.zeros(length)
        for i in range(length):
            sign = 1 if random.random() < 0.5 else -1
            disturbed_weight[i] = self.learner_weights[i] + np.random.exponential(scale) / scale * sign
        index = np.argmin(disturbed_weight)
        return self.recent_guess[index], index

    def fit(self, X, y):
        """
        fit the next batch of training data with a new supervised_learner
        :param X: training features [state, action]
        :param y: training labels [value of state-action function]
        """
        sl = clone(self.regressor)
        self.supervised_learners.append(sl.fit(X, y))
        length = len(self.supervised_learners)
        self.recent_guess = np.zeros(length)
        self.learner_weights = np.zeros(length)
        print('accuracy is : ',sl.score(X,y))
