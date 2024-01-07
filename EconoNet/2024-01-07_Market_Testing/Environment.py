# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 21:39:07 2023

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
        
        self.produce_actions = np.arange(1, self.n_products+1)
        
    def reset(self):
        
        for agent in self.agent_list:
            agent.reset()
        
    def env_step(self, action_dict):
         
         self.reset_market()
         self.reset_flows() # May want to move into the first line of each of the below perform decisions. So we minimizes looping through agents
         self.perform_actions(action_dict) # Here is where q, c, c_error, and Q get updated.
         self.update_consumption_deficit() # This is where D gets updated
         
    def reset_market(self):
        self.market.reset()
    
    def reset_flows(self):
        # Reset flows for each agent
        for Agent in self.agent_list:
                
            Agent.q[:] = 0
            Agent.c[:] = 0
            Agent.c_error[:] = 0

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
            

###############################################################################

    def perform_produce(self, action_dict):
        
        # Grab Producing Agents
        ProduceDict = {n:a for (n,a) in action_dict.items() if a in self.produce_actions}
        
        for (Agent,a) in ProduceDict.items():
              
            Ins = Agent.Ins

            # Choose which sector
            # This should be an extra bit in the decision process of the neural net, and then passed into this function 
            # for now, randomly select
            #allowed_sectors = np.where(Ins.matrix.sum(axis=1) != 0, 1,0)
            #selected_sector = np.random.choice(np.where(allowed_sectors==1)[0])
            selected_sector = a-1 # sector produce decisions start at 1, subtract this off.
                                             
                                               
            u = np.zeros(self.n_products)  
            u[selected_sector] = 1.0 #should this be dt or "1" (or maybe sigma where sigma is the skill-level from 0 to 1)
            
            # Apply dL vector to Instrument
            q = Ins.matrix @ u 
            # Include q into (producing) Agent's production rate
            # q and c defined on half-mesh
            Agent.q = q
            
            # Add this product to the employer's stock
            # Should use RK integration - but where will this be performed, here?
            Agent.Q = Agent.Q + Agent.q*self.dt
            
    def perform_consume(self, action_dict): 

        # Grab Consuming Agents
        ConsumeDict = {n:a for (n,a) in action_dict.items() if a==self.n_products+1}
        
        #print('Consume')
        #print(ConsumeDict)
        
        for Agent in ConsumeDict:
            
            D = Agent.D # consumption defecit
            Q = Agent.Q # agent's product stock  
            
            # Compare D and Q for each product
            # If D > Q, then Q needs to be consumed (can't consume more than Q)
            # if D < Q, then D needs to be consumed
            # BUT, since D can be negative, we need an extra check to ensure that consumption is always positive
            # so if D < 0, no consumption required
            #!!! The if D < 0 do not consume logic could be placed in the Agent's 'choose action' function
            C = np.maximum(D*0 , np.minimum(D, Q)) # this ensures that consumption (C) is always between 0 and Q
            
            # The above is not a rate, but a stock quantity
            # Units are [amount] , not [amount]/[time]
            # q and c defined on half-mesh
            Agent.c = C/self.dt
            
            # Set agent's stock after consumption
            Agent.Q = Agent.Q - C
            
    def perform_exchange(self, action_dict):
        
        # Grab all exchanging agents
        ExchangeDict = {n:a for (n,a) in action_dict.items() if a==self.n_products+2}
        
        # Perform all exchange actions for each agent
        self.market.run_exchange(ExchangeDict)
        

        
