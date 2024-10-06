# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 22:25:25 2023

@author: Alex
"""

import numpy as np

class Environment:
    def __init__(self, dt, agent_list, inst_list, market): # include an action dict that specifies which number goes to which action
        
        
        self.agent_list = agent_list
        self.inst_list = inst_list
        self.market = market
        
        self.dt = dt
        
        self.n_products = self.agent_list[0].Q.shape[0]
        self.action_space = self.agent_list[0].action_space
        
        self.produce_action_number  = 1
        self.consume_action_number  = 2
        self.exchange_action_number = 3
        
        #self.initialize()
                
    def reset(self):
        
        for agent in self.agent_list:
            agent.reset()
            
        self.initialize_bids_asks()
        
    def initialize_bids_asks(self):
        
        for agent in self.agent_list:
            
            agent.p_buy = np.random.uniform(agent.M, size=self.n_products)
            agent.p_sell = np.random.uniform(agent.M, size=self.n_products)
            
    def env_step(self, action_dict):
        
        self.reset_market()
        self.reset_flows() # May want to move into the first line of each of the below perform decisions. So we minimizes looping through agents
        self.perform_actions(action_dict) # Here is where q, c, c_error, and Q get updated.
        self.update_consumption_deficit() # This is where D gets updated
        
    def reset_flows(self):
        # Reset flows for each agent
        for Agent in self.agent_list:
                
            Agent.q[:] = 0
            Agent.c[:] = 0
            Agent.c_error[:] = 0
            
    def reset_market(self):
        self.market.reset()
            
    def perform_actions(self, action_dict):
        
        self.perform_produce(action_dict)
        self.perform_consume(action_dict)
        self.perform_exchange(action_dict)
        
    def update_consumption_deficit(self): # have each input an agent, then go through these in one loop
        
        # Calculate Consumption Defecit
        for Agent in self.agent_list:
            
            # Consumption Error
            Agent.c_error = (Agent.cg - Agent.c) # IF THESE ARE NONE IT MAY NOT WORK
            
            # Integrate c_error to get running consumption defecit
            Agent.D = Agent.D + Agent.c_error*self.dt

        
    ### ACTION PERFORMANCE ###
    ##########################
    
    # Perform all produciton actions for each agent
    
    def perform_produce(self, action_dict):
        #ProduceDict = {n:a for (n,a) in action_dict.items() if a==1}
        ProduceDict = {n:a for (n,a) in action_dict.items() if a==self.produce_action_number}
        
        
        for (Agent,a) in ProduceDict.items():
            
 
            Ins = Agent.Ins
                    
            # Apply dL vector to Instrument
            q = np.copy(Ins.prod_vec)
            # Include q into (producing) Agent's production rate
            # q and c defined on half-mesh
            Agent.q = q
            
            #print(Agent.Q)
            #print(Ins.matrix, u)
            #print(Agent.q)
            
            
            # Add this product to the employer's stock
            # Should use RK integration - but where will this be performed, here?
            #!!! Using an <Agent>.Q attribute instead of a <ProductStock> class for Q 
            #EmployerAgent.Q += q*self.dt 
            #Agent.Q += Agent.q*self.dt
            Agent.Q = Agent.Q + Agent.q*self.dt
            
            #print(Agent.Q)
            #print()
            
    def perform_consume(self, action_dict): 
        # Perform all consumption actions for each agent
    
        # Grab all the Consuming Agents
        
        ConsumeDict = {n:a for (n,a) in action_dict.items() if a==self.consume_action_number}
        
        for Agent in ConsumeDict:
            
            #print(Agent)
            
            D = Agent.D # consumption defecit
            Q = Agent.Q # agent's product stock  
            
            # Compare D and Q for each product
            # If D > Q, then Q needs to be consumed (can't consume more than Q)
            # if D < Q, then D needs to be consumed
            # BUT, since D can be negative, we need an extra check to ensure that consumption is always positive
            # so if D < 0, no consumption required
            #!!! The if D < 0 do not consume logic could be placed in the Agent's 'choose action' function
            C = np.maximum(0 , np.minimum(D, Q)) # this ensures that consumption (C) is always between 0 and Q
            
            # The above is not a rate, but a stock quantity
            # Units are [amount] , not [amount]/[time]
            # q and c defined on half-mesh
            Agent.c = C/self.dt
            
            #print(Agent.Q, Agent.D)
            #print(C)
            
            # Set agent's stock after consumption
            #Agent.Q -= C
            Agent.Q = Agent.Q - C
            
            #print(Agent.Q)
            #print()
            
    def perform_exchange(self, action_dict):
        
        # Grab all exchanging agents
        ExchangeDict = {n:a for (n,a) in action_dict.items() if a==self.exchange_action_number}
        
        # Perform all exchange actions for each agent
        if len(ExchangeDict) > 0:
            money_exchanged, quantity_exchanged, n_tries = self.market.run_exchange(ExchangeDict)
        
