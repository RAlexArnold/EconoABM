# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:45:05 2023

@author: Alex
"""

import torch as T

import numpy as np 

class AbstractAgent():

    def __init__(self):
        pass
    
class Agent(AbstractAgent):
    def __init__(self, index, QNN, Q=None, D=None, M=None, cg=None, n_actions=None):
        
        super().__init__()
        
        self.index = index
        
        self.Q_eval = QNN
        
        self.Q = Q
        self.Q0 = Q
        self.D = D 
        self.D0 = D
        self.M = M
        self.M0 = M
        
        self.observation = None

        self.cg = cg
        self.cg0 = cg
        
        self.action_space = [i for i in range(n_actions)]
        
        if self.Q is None:
            self.c = None 
            self.q = None 
            self.c_error = None
            
        else:
            zeros = np.zeros(self.Q.shape[0])
            self.c = zeros
            self.q = zeros
            self.c_error = zeros
            
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
            
    def choose_action(self, observation):
        if np.random.random() > self.Q_eval.epsilon:
            #state = T.tensor([observation]).to(self.Q_eval.device) # Take our observation and turn it into a PyTorch tensor, and send it to the device because network lives on device.
            
            #state = T.tensor([observation], dtype=T.float32).to(self.Q_eval.device) # this avoids type errors
            
            # The above will work, but gives a notice on speed
            
            ###
            # C:\Users\Alex\Research\EconoNet\DeepProxyEnvironment\DeepQNetworkAgent.py:80: 
            # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. 
            # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. 
            # (Triggered internally at C:\cb\pytorch_1000000000000\work\torch\csrc\utils\tensor_new.cpp:248.)
            # state = T.tensor([observation], dtype=T.float32).to(self.Q_eval.device)
            
            # So I will replace the above with the below, assuming observation will be a numpy array
            state = T.tensor(observation, dtype=T.float32).to(self.Q_eval.device) # this avoids type errors
            
            
            
            actions = self.Q_eval.forward(state) # Send the state to the NN and have it output the actions
            action = T.argmax(actions).item()
            
        else:
            action = np.random.choice(self.action_space)
            
        return action