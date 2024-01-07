# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 00:36:53 2023

@author: Alex
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import Markets


Nproducts = 2

class Agent:
    def __init__(self):
        self.Q = np.random.rand(Nproducts)*10
        self.D = np.zeros(Nproducts)+5
        

N = 5

ExchangeDict = {}
for i in range(N):
    
    agent_i = Agent()
    
    ExchangeDict[agent_i] = 1
    
AgentList = list(ExchangeDict.keys())
    
    
fridge = Markets.ProductBank(Nproducts=Nproducts)

# Create the dictionary of agents and their intended exchanges
fridge.IntendedExchange = fridge.set_intended_exchange(ExchangeDict)

# Initialize the attempts per agent
fridge.AgentAttempts = fridge.set_agent_attempts(ExchangeDict)

def step():
    
    print(fridge.IntendedExchange)
    print()
    fridge.attempt_exchange()
    

def test_steps(T):

    Q_df = np.zeros((T, Nproducts, N))
    stock_df = np.zeros((T, Nproducts))
    
    for t in range(T):
        stock_df[t] = fridge.stock
        for j in range(Nproducts):   
            for v, agent in enumerate(AgentList):
                Qj_agent = agent.Q[j]
                Q_df[t, j, v] = Qj_agent
                
        fridge.attempt_exchange()
        
    f, ax = plt.subplots(Nproducts, figsize=(8, Nproducts*4))
    
    for j in range(Nproducts):
        
        product_df = pd.DataFrame(Q_df[:,j,:])
        fridge_df = pd.Series(stock_df[:,j])
        
        sns.lineplot(data=product_df, ax=ax[j])
        sns.lineplot(data=fridge_df, ax=ax[j], color='black', label='fridge')
        ax[j].axhline(5, color='red', ls=':')
        
    plt.tight_layout()
    
T = 10
#test_steps(T)

fridge.run_exchange(ExchangeDict)
    

        
        
    
    