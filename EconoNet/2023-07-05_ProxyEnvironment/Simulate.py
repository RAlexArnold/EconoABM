# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 00:00:43 2023

@author: Alex
"""

import numpy as np

class Simulate():
    
    def __init__(self, agent, env):
        
        self.agent = agent
        self.env = env
        
        self.dt = env.dt
        
        self.Aarray = None
        self.qarray = None
        self.Carray = None
        self.Qarray = None
        self.Darray = None
        self.Rarray = None
        
        self.epsilon_list = None
        self.Parray = None
        
    def update_arrays(self, ti, reward, action):
        
        self.Aarray[ti] = action
        self.qarray[ti] = self.agent.q
        self.Carray[ti] = self.agent.c
        self.Qarray[ti] = self.agent.Q
        self.Darray[ti] = self.agent.D
        self.Rarray[ti] = reward
        
        
    def calculate_probabilities(self):
    
        action_array = self.Aarray[1:]
    
        unique_actions = np.unique(action_array)
        unique_actions = unique_actions[~np.isnan(unique_actions)]
    
        probabilities = []
        for action in unique_actions:
    
            action_TF = np.where(action_array == action, 1, 0)
            running_mean = np.cumsum(action_TF)/np.arange(1, len(action_TF)+1)
    
            probabilities.append(running_mean)
    
        probabilities = np.array(probabilities)
    
        assert probabilities.sum(axis=0).mean() == 1.0
    
        new_column = np.zeros((self.env.n_actions, 1))*np.nan
        probabilities = np.hstack((new_column, probabilities))
    
        return probabilities
    
    def exponential_weighted_average(self, X, alpha):
        
        n = len(X)
        
        weights = np.exp(alpha * np.arange(n))
        weights /= weights.sum()
        
        return np.mean(X, weights=weights)
        
        
    def run_simulation(self, Ntimes):
        
        n = self.env.n_products
        
        t = 0

        tmax = self.dt*Ntimes
        self.trange = np.arange(t,tmax, self.dt)
        
        self.Aarray = np.zeros(Ntimes)
        self.qarray = np.zeros(shape=(Ntimes, n))
        self.Carray = np.zeros(shape=(Ntimes, n))
        self.Qarray = np.zeros(shape=(Ntimes, n))
        self.Darray = np.zeros(shape=(Ntimes, n))
        self.Rarray = np.zeros(Ntimes)
        
        self.env.reset()
        
        action = np.nan
        reward = np.nan #env.update_reward()
        observation = self.env.observe_state()
        self.epsilon_list = []
        
        for ti, t in enumerate(self.trange):
    
            self.update_arrays(ti, reward, action)
            self.epsilon_list.append(self.agent.epsilon)
            
            #print(observation)
            
            # Choose Action
            action = self.agent.choose_action(observation)
        
            observation_, reward = self.env.step(action)
        
            self.agent.store_transition(observation, action, reward, observation_, False)
        
            self.agent.learn()
        
            observation = observation_
            
        self.epsilon_list = np.array(self.epsilon_list)
        self.Parray = self.calculate_probabilities()
        
        
        return self.trange, self.Aarray, self.qarray, self.Carray, self.Qarray, self.Darray, self.Rarray, self.epsilon_list, self.Parray
                