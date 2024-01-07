# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 01:52:07 2023

@author: Alex
"""

import numpy as np

class Simulate():
    
    def __init__(self, env):
        
        self.env = env
        self.market = self.env.market
        
        self.n = self.env.n_products
        self.n_agents = len(self.env.agent_list)
        
        self.agent_list = self.env.agent_list
        self.inst_list = self.env.inst_list
        #self.dist_list = self.env.dist_list (market)
        
        self.dt = env.dt
        
        self.state_dict  = dict(zip(self.agent_list, [np.nan]*self.n_agents))
        self.action_dict = dict(zip(self.agent_list, [np.nan]*self.n_agents))
        self.reward_dict = dict(zip(self.agent_list, [np.nan]*self.n_agents))
        
        self.Aarray = None
        self.qarray = None
        self.carray = None
        self.Qarray = None
        self.Darray = None
        self.Rarray = None
        
        self.Marray = None
        
        self.epsilon_list = None
        self.Parray = None
        
        self.reward_func = None
        
    def reset(self):
        self.env.reset()
        self.state_dict  = dict(zip(self.agent_list, [np.nan]*self.n_agents))
        self.action_dict = dict(zip(self.agent_list, [np.nan]*self.n_agents))
        self.reward_dict = dict(zip(self.agent_list, [np.nan]*self.n_agents))
        
        self.Aarray = None
        self.qarray = None
        self.carray = None
        self.Qarray = None
        self.Darray = None
        self.Rarray = None
        
        self.Marray = None
        
        self.epsilon_list = None
        self.Parray = None
        
        self.reward_func = None
        
    def set_reward_function(self, reward_func):
        # Method to set the desired reward function
        # 'error' for using consumption error
        # 'deficit' for using the consumption deficit
        
        self.reward_func = reward_func
        
    def reward_function_consumptionError(self, agent):
        
        reward = - np.sum(agent.c_error)
        
        return reward
    
    def reward_function_deficit(self, agent):
        
        reward =  - np.sum(agent.D)
        
        return reward
    
    
    def initialize_arrays(self, Ntimes):
        
        self.Aarray = np.zeros(shape=(Ntimes, self.n_agents))
        self.qarray = np.zeros(shape=(Ntimes, self.n, self.n_agents))
        self.carray = np.zeros(shape=(Ntimes, self.n, self.n_agents))
        self.Qarray = np.zeros(shape=(Ntimes, self.n, self.n_agents))
        self.Darray = np.zeros(shape=(Ntimes, self.n, self.n_agents))
        self.Rarray = np.zeros(shape=(Ntimes, self.n_agents))
        
        self.Marray = np.zeros(shape=(Ntimes, self.n_agents))
        
        self.narray = np.zeros(shape=(Ntimes))
        self.StockArray = np.zeros(shape=(Ntimes, self.n))
        self.DepositsArray = np.zeros(shape=(Ntimes, self.n))
        self.WithdrawalsArray = np.zeros(shape=(Ntimes, self.n))
        self.mEarray = np.zeros(shape=(Ntimes, self.n))
        self.mMarray = np.zeros(shape=(Ntimes, self.n))
        self.parray = np.zeros(shape=(Ntimes, self.n))
        
    def exponential_weighted_average(self, X, alpha):
        
        n = len(X)
        
        weights = np.exp(alpha * np.arange(n))
        weights /= weights.sum()
        
        return np.mean(X, weights=weights)
    
###############################################################################
    def run_simulation(self, Ntimes):
        
        self.reset()
        self.initialize_arrays(Ntimes)
        
        t = 0
        tmax = self.dt*Ntimes
        self.trange = np.arange(t,tmax, self.dt)
        
        #action = np.nan
        #reward = np.nan #env.update_reward()
        #observation = self.env.observe_state() # this needs to be done for each agent separately
        #self.epsilon_list = [] # Needs to be Nagent dimensional
        
        for ti,t in enumerate(self.trange):
            
            self.sim_step(ti)
            
    def sim_step(self, ti):
        
        self.update_market_arrays(ti)
        
        for agent_index, agent in enumerate(self.agent_list):
            
            self.update_arrays(ti, agent_index, agent)
            
            # Get agent's state
            observation = self.get_state(agent)
            
            # Choose action
            action = agent.choose_action(observation)
            
            # Store this in class
            self.state_dict[agent] = observation
            self.action_dict[agent] = action
            
        # Perform all economic actions
        self.env.env_step(self.action_dict)
        
        # Now find reward, new observatoin, and teach the agents
        for agent in self.agent_list:
            
            observation = self.state_dict[agent]
            action = self.action_dict[agent]
            
            observation_ = self.get_state(agent)
            reward = self.update_reward(agent)
            
            agent.store_transition(observation, action, reward, observation_, False)
            agent.learn() # if this is really slow, then do learn with some probability?
            
    def update_market_arrays(self, ti):
        
        self.narray[ti] = self.market.n_tries
        self.StockArray[ti] = self.market.stock
        self.DepositsArray[ti] = self.market.total_deposits
        self.WithdrawalsArray[ti] = self.market.total_withdrawals
          
    def update_arrays(self, ti, agent_index, agent):
        
        self.Aarray[ti, agent_index] = self.action_dict[agent] #action # self.agent_dict
        self.qarray[ti,:,agent_index] = agent.q
        self.carray[ti,:,agent_index] = agent.c
        self.Qarray[ti,:,agent_index] = agent.Q
        self.Darray[ti,:,agent_index] = agent.D
        self.Rarray[ti,agent_index] = self.reward_dict[agent] #reward # self.reward_dict
        
            
    def get_state(self, agent):
        
        Xi_column_sums = agent.Ins.matrix.sum(axis=0)
    
        observation = np.concatenate([agent.Q, agent.D, [agent.M], Xi_column_sums])

        return observation
    
    def update_reward(self, agent):
        
        # Calculate Reward
        # still confusion about where reward structure vs economic structure will go
        # reward functions are here in the environment, but also used in the simulation class
      
        if self.reward_func == 'error':
            reward = self.reward_function_consumptionError(agent)
            
        elif self.reward_func == 'deficit':
            reward = self.reward_function_deficit(agent)
            
        else:
            reward = self.reward_function_deficit(agent)     

        return reward

        
        