from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
import numpy as np
from collections import deque


class RWM:
    """
    randomized weighted majority
    """

    def __init__(self, beta = 0.995):
        """
        initialize an instance of randomized weighted majority learner
        :param beta: [0.8], penalty for wrong guess
        """
        self.supervised_learners = deque()
        self.weight = None
        self.probability = None
        self.previous_guess = None
        self.beta = beta

    def adjust_weight(self, best_possible_action):
        """
        adjust the weight and probability to follow from the best possible action
        :param best_possible_action: int action
        """
        # if there is no leanrner or one learner
        if self.weight is None or len(self.weight) == 1:
            return

        # adjust weight
        for i in range(len(self.previous_guess)):
            if self.previous_guess[i] != best_possible_action:
                self.weight[i] = self.weight[i] * self.beta
        #if max(self.weight)/min(self.weight) > 1e2:
        #    self.weight = np.ones(len(self.weight))
        if min(self.weight) < 1e-8:
            self.weight *= 1e8
        # adjust probability
        total_probability = sum(self.weight)
        self.probability = self.weight / total_probability

    def qval(self, state, one_action):
        """
        compute q values for state, action with randomized weighted majority
        :param state: a list of values
        :param one_action: one action
        :return the value for state-action function
        """
        if not self.supervised_learners:
            return 0.

        x = np.r_[state, one_action].reshape((1,-1))
        learner_number = np.arange(len(self.supervised_learners))
        learner_id = np.random.choice(learner_number, p=self.probability)
        return self.supervised_learners[learner_id].predict(x)

    def predict(self, state, possible_actions):
        """
        give action based on current state
        :param state: a list of values
        :param possible_actions: a list of possible actions
        :return the action to maximize the state-action function
        """
        if not self.supervised_learners:
            return np.random.choice(possible_actions)
        self.previous_guess = np.zeros(len(self.supervised_learners))
        for i in range(len(self.supervised_learners)):
            state_action_values = np.zeros(len(possible_actions))
            for j in range(len(possible_actions)):
                x = np.r_[state, possible_actions[j]].reshape((1,-1))
                state_action_values[j] = self.supervised_learners[i].predict(x)
            self.previous_guess[i] = possible_actions[np.argmax(state_action_values)]
        learner_list = np.arange(len(self.supervised_learners))
        learner_id = np.random.choice(learner_list, p = self.probability)
        return self.previous_guess[learner_id]

    def fit(self, X, y):
        """
        fit the next batch of training data with a new supervised_learner
        :param X: training features [state, action]
        :param y: training labels [value of state-action function]
        """
        sl = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=6,
                                                learning_rate = 0.01, loss='ls', min_samples_split=2)
        self.supervised_learners.append(sl.fit(X, y))
        if len(self.supervised_learners)>30:
            self.supervised_learners.popleft()
        self.weight = np.ones(len(self.supervised_learners))
        self.probability = self.weight/(sum(self.weight))
        print('accuracy is : ',sl.score(X,y))
