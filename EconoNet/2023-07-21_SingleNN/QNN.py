# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:47:29 2023

@author: Alex
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 


#input_dims = 3*Nagents + 1

class DeepQNetwork(nn.Module):
    
    def __init__(self, 
                 input_dims,
                 n_actions,
                 n_agents,
                 fc1_dims=256, 
                 fc2_dims=256, 
                 gamma=0.9, 
                 epsilon=1.0, 
                 lr=0.001, 
                 batch_size=100, 
                 max_mem_size=10_000, 
                 eps_end=0.001, 
                 eps_dec=0.01,
                 ):
        
        super(DeepQNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.n_agents = n_agents
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions  = n_actions
        
        self.gamma = gamma
        
        self.epsilon0 = epsilon
        
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        
        self.initialize()
        
    
    def initialize(self):
        
        self.epsilon = self.epsilon0
        
        self.mem_cntr = 0
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # Input (first) layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) # Second layer
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions) # Output (third) layer
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr) # Adam optimizer. Stochastic Gradient Descent using moments
        self.loss = nn.MSELoss() # Mean Square Error Loss Function
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # Will default to CPU
        self.to(self.device)
        
        self.state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32) 
         
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32) # won't work if we want to include labor vector one hot encoder in action-output?
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
    def reset(self):
        self.inititialize()
        

        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        
        return actions
    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def learn(self):
        if self.mem_cntr < self.batch_size: # If memory counter less than batch size, just return. No learning
        # Learn function is reiteration of game loop. But for continuious learning I may need to change this??
            return
        
        #print('Begin Learning')
        #print(f'mem_cntr: {self.mem_cntr}, batch_size: {self.batch_size}')
        
        self.optimizer.zero_grad()
        
        max_mem = min(self.mem_cntr, self.mem_size)
        
        #print(f'max_mem: {max_mem}')
        
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        #print(f'batch: {batch, len(batch)}')
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = T.tensor(self.state_memory[batch]).to(self.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.device)
        
        action_batch = self.action_memory[batch]
        
        q_eval = self.forward(state_batch)[batch_index, action_batch] #value of actions we took
        q_next = self.forward(new_state_batch) 
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        
        loss = self.loss(q_target, q_eval).to(self.device)
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                        else self.eps_min