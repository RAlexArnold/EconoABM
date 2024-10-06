# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:00:35 2024

@author: Alex
"""
import numpy as np
import Agents
import MarketsV03 as Markets

class PriceRatioExperiment:
    
    def __init__(self, N, l1,l2,c1,c2, *, Mmax=100.0, theta=0.1, epsilon=0.005, max_tries=100, initial_money = 'random'):
        
        self.N = N
        self.l1 = l1
        self.l2 = l2
        self.c1 = c1
        self.c2 = c2
        
        
        self.Mmax = Mmax
        self.theta = theta
        self.epsilon = epsilon
        self.max_tries = max_tries
        
        # Determine labor sector sizes
        self.N1 = int(round(l1*c1*N, 0))
        self.N2 = self.N - self.N1
        #self.N2 = int(round(l2*c2*N, 0))
        
        self.q1 = np.array([1.0/l1, 0.0])
        self.q2 = np.array([0.0   , 1.0/l2])
        
        self.c  = np.array([c1, c2])
        
        self.Q1 = np.array([0.0, 0.0])
        self.Q2 = np.array([0.0, 0.0])
        
        self.D1 = self.c.copy()
        self.D2 = self.c.copy()
        
        if initial_money == 'random':
            self.agent1_list = [Agents.Agent(self.Q1.copy(), self.D1.copy(), np.random.uniform(self.Mmax), n_actions=1, input_dims=[1]) for _ in range(self.N1)]
            self.agent2_list = [Agents.Agent(self.Q2.copy(), self.D2.copy(), np.random.uniform(self.Mmax), n_actions=1, input_dims=[1]) for _ in range(self.N2)]
            
        elif initial_money == 'uniform':
            self.agent1_list = [Agents.Agent(self.Q1.copy(), self.D1.copy(), self.Mmax, n_actions=1, input_dims=[1]) for _ in range(self.N1)]
            self.agent2_list = [Agents.Agent(self.Q2.copy(), self.D2.copy(), self.Mmax, n_actions=1, input_dims=[1]) for _ in range(self.N2)]
            

        self.agent_list = self.agent1_list + self.agent2_list
        
        self.n = 2 # 2 commodities
        
        self.initialize_bids_asks()
        self.initialize_market()
        
        
    def initialize_bids_asks(self):
        
        for agent in self.agent_list:
            
            agent.p_buy = np.random.uniform(agent.M, size=self.n)
            agent.p_sell = np.random.uniform(agent.M, size=self.n)
            
    def initialize_market(self):
        
        self.scda = Markets.SCDA(self.n, theta=self.theta, gamma=1.0, epsilon=self.epsilon, max_tries=self.max_tries, M_min=0.01, peval='log_mean')
        self.ExchangeDict = {agent : 2 for agent in self.agent_list}
        
    def update_agent_state(self):
        
        for agent1 in self.agent1_list:
            
            c_t = np.minimum(agent1.Q, agent1.D)
            agent1.D = agent1.D + (self.c - c_t)
            agent1.Q = np.maximum(0, agent1.Q + self.q1 - c_t)
            
        for agent2 in self.agent2_list:
            
            c_t = np.minimum(agent2.Q, agent2.D)
            agent2.D = agent2.D + (self.c - c_t)
            agent2.Q = np.maximum(0, agent2.Q + self.q2 - c_t)
            
        
    def run_experiment(self, T):
        
        # For market
        self.Me_df = np.zeros((T-1,self.n))
        self.Qe_df = np.zeros((T-1,self.n))
        self.ntries_df = []
        
        # For agents
        self.Magent_df = np.zeros((T-1, self.N))
        
        for t in range(T-1):
            
            Me, Qe, ntries = self.scda.run_exchange(self.ExchangeDict)
            
            self.update_agent_state()
            
            self.Me_df[t] = Me
            self.Qe_df[t] = Qe
            self.ntries_df.append(ntries)
            
            self.Magent_df[t] = self.scda.agent_Mdata
            
        self.p_df = self.Me_df/self.Qe_df
        
        return self.Me_df, self.Qe_df, self.ntries_df, self.p_df, self.Magent_df
        
            
        
        
        
        
        
        
        
        