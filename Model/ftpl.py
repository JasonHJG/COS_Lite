from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
import numpy as np
from collections import deque


class FTPL:
    """
    follow the perturbed leader
    """

    def __init__(self, eps = 0.05, limit = 20):
        """
        initialize an instance of randomized weighted majority learner
        """
        self.supervised_learners = deque()
        self.weight = None
        self.probability = None
        self.previous_guess = None
        self.eps = eps

    def adjust_weight(self, utility_array):
        """
        adjust the weight and probability to follow from the best possible action
        :param best_possible_action: int action
        """
        # if there is no leanrner or one learner
        if self.weight is None or len(self.weight) == 1:
            return

        # adjust weight, now weight is associated with penalty
        mean_performance = np.mean(utility_array)
        for i in range(len(self.previous_guess)):
            if utility_array[i] <= mean_performance:
                self.weight[i] += np.abs(utility_array[i])

    def qval(self, state, one_action):
        """
        compute q values for state, action with follow the perturbed leader
        :param state: a list of values
        :param one_action: one action
        :return the value for state-action function
        """
        if not self.supervised_learners:
            return 0.

        x = np.r_[state, one_action].reshape((1,-1))
        perturbed_w = self.weight + np.random.uniform(0, 1/self.eps, len(self.weight))
        #learner_id = np.argmin(perturbed_w)
        # todo: test if use top 25% is an good idea
        learner_id = np.argsort(perturbed_w)
        q_value = 0
        for i in range(int(np.ceil(0.5 * len(learner_id)))):
            q_value += self.supervised_learners[learner_id[i]].predict(x)
        return q_value/int(np.ceil(0.5 * len(learner_id)))

        #return self.supervised_learners[learner_id].predict(x)

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
        perturbed_w = self.weight + np.random.uniform(0, 1 / self.eps, len(self.weight))
        learner_id = np.argsort(perturbed_w)
        # todo: possible modification
        # todo: average the top 25%'s q function

        q_value_array = np.zeros(len(possible_actions))
        for i in range(len(possible_actions)):
            x = np.r_[state, possible_actions[i]].reshape((1, -1))
            for j in range(int(np.ceil(0.5 * len(learner_id)))):
                q_value_array[i] += self.supervised_learners[learner_id[j]].predict(x)
        return possible_actions[np.argmax(q_value_array)]
        """
        action = 0
        for i in range(int(np.ceil(0.25 * len(learner_id)))):
            action += self.previous_guess[learner_id[i]]
        return action /  int(np.ceil(0.25 * len(learner_id)))
        """


    def fit(self, X, y):
        """
        fit the next batch of training data with a new supervised_learner
        :param X: training features [state, action]
        :param y: training labels [value of state-action function]
        """

        if len(self.supervised_learners)>=15:
            print('enough ftpl learners, stop training')
            self.weight = np.ones(len(self.supervised_learners))
            self.probability = self.weight / (sum(self.weight))
        else:

            sl = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=6,
                                                learning_rate = 0.01, loss='ls', min_samples_split=2)
            self.supervised_learners.append(sl.fit(X, y))
            self.weight = np.ones(len(self.supervised_learners))
            self.active_weight = self.weight
            print('accuracy is : ',sl.score(X,y))
