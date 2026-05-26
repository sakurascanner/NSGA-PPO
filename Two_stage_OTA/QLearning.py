import gymnasium as gym
import numpy as np

class QLearning:
    def __init__(self, state_dim, action_dim, learning_rate=0.12, gamma=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 0.9
        self.pi = np.full((self.state_dim,self.action_dim),1/action_dim)
        self.q_table = np.zeros((self.state_dim,self.action_dim))

    def choose_action(self, state):
        rand_num = np.random.random()
        prob = 0
        action = 0
        for i in range(self.action_dim):
            prob += self.pi[state,i]
            if rand_num < prob:
                action = i
                break
        return action

    def update(self, state, action, reward, next_state, done):
        max_next_state_q = self.q_table[next_state].max()
        updated_state_q = self.q_table[state,action]
        self.q_table[state,action] = updated_state_q+self.learning_rate*(reward+self.gamma*max_next_state_q-updated_state_q)
        for i in range(self.action_dim):
            if max_next_state_q == self.q_table[next_state,i]:
                self.pi[state,i] = self.epsilon
            else:
                self.pi[state,i] = (1-self.epsilon)/(self.action_dim-1)
        
