# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 22:25:25 2023

@author: Alex
"""

import numpy as np

class Environment:
    def __init__(self, dt, agent_list, inst_list, dist_list): # include an action dict that specifies which number goes to which action
        
        
        self.agent_list = agent_list
        self.inst_list = inst_list
        self.dist_list = dist_list
        
        #self.action_dict = None
        #self.reward_dict 
        
        #self.n_actions = n_actions
        self.dt = dt
        #self.productivity = productivity
        
        #self.Q0 = Q0 <- defined when initializing agent
        #self.D0 = D0
        #self.M0 = M0
        
        #self.n_products = self.cg.shape[0]
    
        
        self.n_products = self.agent_list[0].Q.shape[0]
        self.action_space = self.agent_list[0].action_space
        
        self.produce_actions = np.arange(1, self.n_products+1)
        
        #self.initialize()
        
    def reset(self):
        
        for agent in self.agent_list:
            agent.reset()
            
    def env_step(self, action_dict):
        
        self.reset_flows() # May want to move into the first line of each of the below perform decisions. So we minimizes looping through agents
        self.perform_actions(action_dict) # Here is where q, c, c_error, and Q get updated.
        self.update_consumption_deficit() # This is where D gets updated
        #reward = self.update_reward() # This is where the reward is calculated from the action
        #observation = self.observe_state() # This is the state after the action has been taken
        
        #return observation, reward
    
    # Self Independent
    ##################
    
    ### ACTION PERFORMANCE ###
    ### Action Performance ###
    ##########################
    
    # Perform all produciton actions for each agent
    
    def perform_produce(self, action_dict):
        #ProduceDict = {n:a for (n,a) in action_dict.items() if a==1}
        ProduceDict = {n:a for (n,a) in action_dict.items() if a in self.produce_actions}
        
        #print('Produce')
        #print(ProduceDict)
        
        for (Agent,a) in ProduceDict.items():
            
            #print(Agent)
            # Grab employer (should be agent_index)
            #employer_index = Agent.employer_index 
            
            # Above Can move out of loop so different employer_indicies can have different production rules
            # And change to check (if == ). Must then initialize employer_index somewhere else. Or drop it for now
            
            #EmployerAgent = Agent #self.Agent_list[employer_index] #could Agent have an agent object attribute, it's more direct
            
            
            # The above is just setting itself to be the owner of the means of produciton, and the "employer" of themselves
            # Just is premature generalization. But if we keep it, we can move the initialization elswhere.
            
            #Ins = EmployerAgent.Ins  
            Ins = Agent.Ins
            #print(Ins)
            
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
        
        ConsumeDict = {n:a for (n,a) in action_dict.items() if a==self.n_products+1}
        
        #print('Consume')
        #print(ConsumeDict)
        
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
            C = np.maximum(D*0 , np.minimum(D, Q)) # this ensures that consumption (C) is always between 0 and Q
            
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
        # Perform all exchange actions for each agent
        
        # Each agent chooses a market. or for now assign Market to each agent, or assign agent to market...
        # for agent
        # OR for market
        #        for agents in market
        pass
        
    def update_consumption_deficit(self): # have each input an agent, then go through these in one loop
        # Calculate Consumption Defecit
        for Agent in self.agent_list:
            
            # Consumption Error
            Agent.c_error = (Agent.cg - Agent.c) # IF THESE ARE NONE IT MAY NOT WORK
            
            # Integrate c_error to get running consumption defecit
            
            #Agent.D += Agent.c_error*self.dt  should work, but the in-place assignements bugged for the perform_X functions
            Agent.D = Agent.D + Agent.c_error*self.dt
            
            
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
        
        
        


'''
    the reset in the environment should reset the agents to where they initially were
    so either include those in the agent, instrument, etc., or 
'''

'''        
    def initialize(self):
           
        if self.Q0 is None:
            self.Q0 = np.zeros(self.n_products)
        
        if self.D0 is None:
            self.D0 = np.zeros(self.n_products)
            
        self.agent.cg = self.cg
        self.agent.c = np.zeros(self.n_products)
        self.agent.q = np.zeros(self.n_products)
        self.agent.c_error = self.agent.cg - self.agent.c
'''
        