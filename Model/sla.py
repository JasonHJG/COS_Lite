from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
import numpy as np


class SLA:
    """
    Supervised Learner Averaging
    """

    def __init__(self):
        """
        initialize an instance of COS learner
        """
        self.supervised_learners = []

    def qval(self, state, action):
        """
        compute q values for state, action
        :param state: a list of values
        :param action: int, some integer value
        :return the value for state-action function
        """
        if not self.supervised_learners:
            return 0.
        else:
            x = np.r_[state,action].reshape((1,-1))
            return np.mean([sl.predict(x) for sl in self.supervised_learners])

    def predict(self, state, possible_actions):
        """
        give action based on current state
        :param state: a list of values
        :param possible_actions: a list of possible actions
        :return the action to maximize the state-action function
        """
        state_action_values = np.zeros(len(possible_actions))
        for i in range(len(possible_actions)):
            state_action_values[i] = self.qval(state, possible_actions[i])
        return possible_actions[np.argmax(state_action_values)]

    def fit(self, X, y):
        """
        fit the next batch of training data with a new supervised_learner
        :param X: training features [state, action]
        :param y: training labels [value of state-action function]
        """
        sl = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=6,
                                                learning_rate = 0.01, loss='ls', min_samples_split=2)
        self.supervised_learners.append(sl.fit(X, y))
        print('accuracy is : ',sl.score(X,y))
