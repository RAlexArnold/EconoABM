# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:55:50 2023

@author: Alex
"""

import Agents
import Markets

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.close('all')


n = 2
Q = np.ones(n)*2
Q[1] = 1
D = np.zeros(n)
M = 100
cg = np.ones(n)
cg[1] = 1

N = 100
n_actions = 3



agent_list = [Agents.Agent(np.random.randint(3, size=n),D,M,cg, n_actions=n_actions, input_dims=[2*n]) for agents in range(N)]

ProductDict = {}
for agent in agent_list:
    ProductDict[agent] = np.random.randint(0, n)
    
    agent.p_buy = np.random.uniform(M, size=n)
    agent.p_sell = np.random.uniform(M, size=n)
    
alpha = 0.1# 0.01 #0.1# 0.1
beta = 0.15 #0.02 #0.15
gamma = 0.8
epsilon = 0#0.001#0.005
max_agent_tries = 100#100

market = Markets.Marketv3(n, alpha=alpha, beta=beta, gamma=gamma, epsilon=epsilon, max_agent_tries=max_agent_tries)



####
dt = 0.1

def step():
    action_dict = {}
    Q = np.zeros(n)
    D = np.zeros(n)
    
    if np.random.rand() < 0.005:
        i = np.random.randint(n)
        for agent in agent_list:
            agent.Q[i] = agent.Q[i]*(1-0.9)
    
    for agent in agent_list:
        
        agent_action = np.random.randint(0, n_actions) # agent.choose_action(np.random.rand(n))
        
        #agent_action - agent.choose_action(np.concatenate([Q,D]))
        
        action_dict[agent] = agent_action
        Q = Q + agent.Q
        D = D + agent.D        
    
    ProduceDict   = {n:a for (n,a) in action_dict.items() if a==0}
    ConsumeDict   = {n:a for (n,a) in action_dict.items() if a==1}
    ExchangeDict  = {n:a for (n,a) in action_dict.items() if a==2}
    
    for agent in ProduceDict:
        
        i = ProductDict[agent]
        if agent.Q[i] < 20:
            agent.Q[i] = agent.Q[i] + 3*dt
        
    for agent in ConsumeDict:
        
        dC = np.minimum(agent.Q, np.maximum(np.zeros(n), agent.D))
        agent.Q = agent.Q - dC
        agent.Q = np.where(agent.Q < 0, 0, agent.Q)
        agent.D = agent.D + cg*dt - dC 
        
    
    Me, Qe, s2, n_tries = market.run_exchange(ExchangeDict)
    p = Me/Qe
        
    return len(ProduceDict), len(ConsumeDict), len(ExchangeDict), Q, D, p, s2, Me, Qe, n_tries

'''
def learn_step():
    action_dict = {}
    Q = np.zeros(n)
    D = np.zeros(n)
    
    if np.random.rand() < 0.005:
        i = np.random.randint(n)
        for agent in agent_list:
            agent.Q[i] = agent.Q[i]*(1-0.9)
            
    for agent_index, agent in enumerate(self.agent_list):
        
        observation = get_state(agent)
        
        Q = Q + agent.Q
        D = D + agent.D 
        
        # Choose action
        action = agent.choose_action(observation)
        
        # Store this in class
        state_dict[agent] = observation
        action_dict[agent] = action
        
    for agent in ProduceDict:
        
        i = ProductDict[agent]
        if agent.Q[i] < 200:
            agent.Q[i] = agent.Q[i] + 3*dt
        
    for agent in ConsumeDict:
        
        dC = np.minimum(agent.Q, np.maximum(np.zeros(n), agent.D))
        agent.Q = agent.Q - dC
        agent.Q = np.where(agent.Q < 0, 0, agent.Q)
        agent.D = agent.D + cg*dt - dC 
        
    
    Me, Qe, s2, n_tries = market.run_exchange(ExchangeDict)
    p = Me/Qe
    
    # Now find reward, new observatoin, and teach the agents
    for agent in self.agent_list:
        
        observation = state_dict[agent]
        action = action_dict[agent]
        
        observation_ = get_state(agent)
        reward = update_reward(agent)
        
        agent.store_transition(observation, action, reward, observation_, False)
        agent.learn() # if this is really slow, then do learn with some probability?

        
    
    ProduceDict   = {n:a for (n,a) in action_dict.items() if a==0}
    ConsumeDict   = {n:a for (n,a) in action_dict.items() if a==1}
    ExchangeDict  = {n:a for (n,a) in action_dict.items() if a==2}
    

        
   

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
'''

print('Mean Product Distribution')
dist = np.unique([i for i in ProductDict.values()], return_counts=True)[1]

import scipy.stats

ent = scipy.stats.entropy(dist, base=2)
print(ent)
print()


T = 1000
trange = np.arange(0, T+dt, dt)

Tlen = len(trange)
Q = np.zeros((Tlen, n))
D = np.zeros((Tlen, n))
p = np.zeros((Tlen, n))
pstd = np.zeros((Tlen,n))
Me = np.zeros((Tlen,n))
Qe = np.zeros((Tlen,n))
en = np.zeros((Tlen,n))

Metot = np.zeros((Tlen))
ap = np.zeros((Tlen))
ac = np.zeros((Tlen))
ae = np.zeros((Tlen))

for i,t in enumerate(trange):
    
    apt, act, aet, Qt, Dt, pt, pstdt, Met, Qet, ent = step()
    
    Q[i] = Qt/N
    D[i] = Dt/N
    p[i] = pt
    pstd[i] = pstdt
    Me[i] = Met
    Qe[i] = Qet
    en[i] = ent
    
    Metott = np.sum(Met)
    Metot[i] = Metott
    
    ap[i] = apt/N
    ac[i] = act/N
    ae[i] = aet/N
    
print('Done...')
print('Plotting...')    
###
plt.close()
lw = 0.5

plt.rcParams['figure.dpi'] = 400
f, ax = plt.subplots(1)
plt.plot(trange, Q[:, 0], lw=lw)
plt.plot(trange, Q[:, 1], lw=lw)
plt.ylabel('Q')
#plt.yscale('log')

f, ax = plt.subplots(1)
plt.plot(trange, D[:, 0], lw=lw)
plt.plot(trange, D[:, 1], lw=lw)
plt.ylabel('D')
#plt.yscale('log')

f, ax = plt.subplots(1)
plt.plot(trange, D[:, 0] - Q[:,0], lw=lw)
plt.plot(trange, D[:, 1] - Q[:,1], lw=lw)
plt.axhline(0)
plt.ylabel('D-Q')

f, axs = plt.subplots(5, figsize=(8, 3*5))
axs[0].plot(trange, en, lw=lw)
axs[0].set_ylabel('Attempts')
axs[0].set_yscale('log')

axs[1].plot(trange, Metot, lw=lw)
axs[1].set_ylabel('$\Sigma |\Delta M|$')
axs[1].set_yscale('log')

for i in range(n):
    axs[2].plot(trange, Me[:,i], lw=lw)
axs[2].set_ylabel('$\Sigma |\Delta M_{i}|$')
axs[2].set_yscale('log')
    
for i in range(n):
    axs[3].plot(trange, Qe[:,i], lw=lw)
axs[3].set_ylabel('$\Sigma |\Delta Q_{i}|$')
axs[3].set_yscale('log')

for i in range(n):
    axs[4].plot(trange, p[:,i], lw=lw)
axs[4].set_ylabel('$|p_{i}|$')
axs[4].set_yscale('log')
#plt.ylim(ymin=0.1, ymax=M)
    
        

        


#for 