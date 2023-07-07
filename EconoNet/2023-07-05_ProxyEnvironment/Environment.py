# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 21:58:27 2023

@author: Alex
"""
import numpy as np

cg = np.array([1.0])
n_actions = 2
dt = 1.0
productivity= 5.0

class Environment():
    def __init__(self, cg, productivity, dt, n_actions, agent, *, Q0 = None):
        
        self.cg = cg
        self.n_actions = n_actions
        self.dt = dt
        self.productivity = productivity
        self.agent = agent
        
        self.Q0 = Q0
        
        self.n_products = self.cg.shape[0]
        
        if self.Q0 is None:
            self.Q0 = np.zeros(self.n_products)
    
        
        self.omega1 = float(self.cg[0]/self.productivity) # The theoretical ideal rate of labor
        
        self.reward_func = None
        
        self.initialize()
    
    def initialize(self):
        
        
        self.agent.Q = self.Q0
        self.agent.D = np.zeros(self.n_products)
        self.agent.cg = self.cg
        self.agent.c = np.zeros(self.n_products)
        self.agent.q = np.zeros(self.n_products)
        self.agent.c_error = self.agent.cg - self.agent.c
        
    def reset(self):
        self.initialize()

    def set_reward_function(self, reward_func):
        # Method to set the desired reward function
        # 'error' for using consumption error
        # 'deficit' for using the consumption deficit
        
        self.reward_func = reward_func
        

    def reward_function_consumptionError(self):
        
        reward = - np.sum(self.agent.c_error)
        
        return reward
    
    def reward_function_deficit(self):
        
        reward =  - np.sum(self.agent.D)
        
        return reward
    
    def produce(self):
        
        u = np.zeros(self.n_products)
        u[0] = 1 # the agent will only produce the first product
        
        self.agent.q = self.productivity*u
        self.agent.Q += self.agent.q*self.dt
        
    def consume(self):
    
        dC = np.maximum(np.zeros(self.n_products), np.minimum(self.agent.Q, self.agent.D))
    
        self.agent.c = dC/self.dt
        self.agent.Q -= dC
        
    def exchange(self):
    
        # The quantity of products that are lacking (+) or in excess (-)
        Q_needed = self.agent.D-self.agent.Q
    
        Q_to_buy  = np.where(Q_needed > 0, Q_needed, 0) # (+)
        Q_to_sell = np.where(Q_needed < 0, Q_needed, 0) # (-)
    
        for i, Qi in enumerate(Q_to_sell):
            if Qi < 0:
    
                efficiency_selling = np.random.uniform(0, self.agent.Q[i]/(-Qi))
    
                Q_sold = efficiency_selling*Qi
    
                self.agent.Q[i] += Q_sold
    
        for i, Qi in enumerate(Q_to_buy):
            if Qi > 0:
    
                efficiency_buying = np.random.exponential(1)
    
                Q_bought = efficiency_buying * Qi
    
                self.agent.Q[i] += Q_bought
                
    #####
                
    def reset_flows(self):
        
        self.agent.q[:] = 0
        self.agent.c[:] = 0
        self.agent.c_error[:] = 0
        
    def perform_action(self, action):
        
        if self.n_actions == 2:
            
            if action == 0:
                self.produce()
                
            elif action == 1:
                self.consume()
                
        elif self.n_actions == 3:
            
            if action == 0:
                pass
            
            elif action == 1:
                self.produce()
                
            elif action == 2:
                self.consume()
                
        elif self.n_actions == 4:
             
            if action == 0:
                pass
            
            elif action == 1:
                self.produce()
             
            elif action == 2:
                self.consume() 
             
            elif action == 3:
                self.exchange()

    def update_deficit(self):
        
        self.agent.c_error = self.cg - self.agent.c
        self.agent.D += self.agent.c_error*self.dt
        
    def update_reward(self):
        
        if self.reward_func == 'error':
            reward = self.reward_function_consumptionError()
            
        elif self.reward_func == 'deficit':
            reward = self.reward_function_deficit()
            
        else:
            reward = self.reward_function_consumptionError()
            
        return reward

    def observe_state(self):
        
        observation = np.concatenate([self.agent.Q, self.agent.D])
        return observation
    
    def step(self, action):
        
        self.reset_flows()
        self.perform_action(action) # Here is where q, c, c_error, and Q get updated.
        self.update_deficit() # This is where D gets updated
        reward = self.update_reward() # This is where the reward is calculated from the action
        observation = self.observe_state() # This is the state after the action has been taken
        
        return observation, reward
        
    
    
            