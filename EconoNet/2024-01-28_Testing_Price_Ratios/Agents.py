# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 20:50:38 2023

@author: Alex
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions  = n_actions
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # Input (first) layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) # Second layer
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions) # Output (third) layer
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr) # Adam optimizer. Stochastic Gradient Descent using moments
        self.loss = nn.MSELoss() # Mean Square Error Loss Function
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # Will default to CPU
        self.to(self.device)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        
        return actions
    
class AbstractAgent():

    def __init__(self):
        pass
    
class Agent(AbstractAgent):
    def __init__(self, Q=None, D=None, M=None, cg=None, gamma=0.9, epsilon=1.0, lr=0.001, input_dims=None, batch_size=100, n_actions=None,
                 max_mem_size=10_000, eps_end=0.001, eps_dec=0.01):
        
        super().__init__()
        
        self.Q = Q
        self.Q0 = np.copy(Q)
        self.D = D 
        self.D0 = np.copy(D)
        self.M = M
        self.M0 = np.copy(M)
        
        self.observation = None

        self.cg = cg
        self.cg0 = np.copy(cg)
        
        if self.Q is None:
            self.c = None 
            self.q = None 
            self.c_error = None
            
        else:
            zeros = np.zeros(self.Q.shape[0])
            self.c = zeros
            self.q = zeros
            self.c_error = zeros
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        
        self.QNN = DeepQNetwork(self.lr, 
                                n_actions=n_actions, 
                                input_dims=input_dims,
                                fc1_dims=256, 
                                fc2_dims=256)

        
        self.state_memory     = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32) 
         
        self.action_memory    = np.zeros(self.mem_size, dtype=np.int32) # won't work if we want to include labor vector one hot encoder in action-output?
        self.reward_memory    = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory  = np.zeros(self.mem_size, dtype=bool)
        
    
    def reset(self):
        
        self.Q = self.Q0
        self.D = self.D0
        self.M = self.M0
        self.cg = self.cg0
        
        if self.Q is None:
            self.c = None 
            self.q = None 
            self.c_error = None
            
        else:
            zeros = np.zeros(self.Q.shape[0])
            self.c = zeros
            self.q = zeros
            self.c_error = zeros
    
    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        
        self.mem_cntr += 1
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            
            state = T.tensor(observation, dtype=T.float32).to(self.QNN.device) # this avoids type errors
            
            actions = self.QNN.forward(state) # Send the state to the NN and have it output the actions
            action = T.argmax(actions).item()
            
        else:
            action = np.random.choice(self.action_space)
            
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size: # If memory counter less than batch size, just return. No learning
            return

        self.QNN.optimizer.zero_grad() # need to be here, or in the initialization
        
        max_mem = min(self.mem_cntr, self.mem_size)
        
        # Pick random batch of memories
        # First pick random batch indices
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        # And set the new batch indices
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        # Pick random memories
        state_batch = T.tensor(self.state_memory[batch]).to(self.QNN.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.QNN.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.QNN.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.QNN.device)
        action_batch = self.action_memory[batch]
        
        q_eval = self.QNN.forward(state_batch)[batch_index, action_batch] #value of actions we took
        q_next = self.QNN.forward(new_state_batch) 
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        
        loss = self.QNN.loss(q_target, q_eval).to(self.QNN.device)
        loss.backward()
        self.QNN.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                        else self.eps_min
                        