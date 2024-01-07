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
import scipy.stats

    

def perturb(agent_list, *, p=0.1, d=0.9, how='each'):
    
    if np.random.rand() < p:

        if how=='all':
            
            i = np.random.randint(n)
            for agent in agent_list:
                agent.Q[i] = agent.Q[i]*(1-d)
                
        elif how=='each':
            
            for agent in agent_list:
                
                i =  np.random.randint(n)
                agent.Q[i] = agent.Q[i]*(1-d)
                
  
def get_action():
    action_dict = {}
    Q = np.zeros(n)
    D = np.zeros(n)
    M = 0
    qtot = np.zeros(n)
    ctot = np.zeros(n)
    
    

    for agent in agent_list:
        
        agent_action = np.random.randint(0, n_actions) # agent.choose_action(np.random.rand(n))
        
        #agent_action - agent.choose_action(np.concatenate([Q,D]))
        
        action_dict[agent] = agent_action
        Q = Q + agent.Q
        D = D + agent.D 
        M = M + agent.M
    
    # Make action sets
    ProduceDict   = {n:a for (n,a) in action_dict.items() if a==0}
    ConsumeDict   = {n:a for (n,a) in action_dict.items() if a==1}
    #ExchangeDict  = {n:a for (n,a) in action_dict.items() if a==2}
    ExchangeDict = {agent_list[0]: 2, agent_list[1]: 2}
    
    return ProduceDict, ConsumeDict, ExchangeDict, Q, D, M, qtot, ctot

def perform_action(ProduceDict, ConsumeDict, ExchangeDict, Q, D, M, qtot, ctot):
    
    # Perform each action
    for agent in ProduceDict:
        
        i = ProductDict[agent]
        #if agent.Q[i] < Qmax:
            
        q = xi[i]
        agent.Q[i] = agent.Q[i] + q*dt
            
        qtot[i] = qtot[i] + q
        
    for agent in ConsumeDict:
        
        dC = np.minimum(agent.Q, np.maximum(np.zeros(n), agent.D))
        agent.Q = agent.Q - dC
        agent.Q = np.where(agent.Q < 0, 0, agent.Q)
        agent.D = agent.D + cg*dt - dC 
        
        ctot = ctot + dC/dt
        
    
    Me, Qe, s2, n_tries = market.run_exchange(ExchangeDict)
    p = Me/Qe
    
    state = [qtot, ctot, Q,D,M, p, s2, Me, Qe, n_tries]
        
    return len(ProduceDict), len(ConsumeDict), len(ExchangeDict), state

def step():
    
    ProduceDict, ConsumeDict, ExchangeDict, Q, D, M, qtot, ctot = get_action()
    
    ap, ac, ae, state = perform_action(ProduceDict, ConsumeDict, ExchangeDict, Q, D, M, qtot, ctot)
    
    return ap, ac, ae, state
    

def main(agent_list, p_perturn, T, dt, d=0.9, how='each'):
    
    trange = np.arange(0, T+dt, dt)
    
    Tlen = len(trange)
    q = np.zeros((Tlen, n))
    c = np.zeros((Tlen, n))
    Q = np.zeros((Tlen, n))
    D = np.zeros((Tlen, n))
    M = np.zeros((Tlen))
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
        
        print()
        print()
        print(f'---- Time {t} ----')
        print()
        print()
        
        perturb(agent_list, p=p_perturb, d=d, how=how)
        apt, act, aet, state = step()
        qt, ct, Qt ,Dt, Mt, pt, pstdt, Met, Qet, ent = state
        
        q[i] = qt/N
        c[i] = ct/N
        Q[i] = Qt/N
        D[i] = Dt/N
        M[i] = Mt
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
        
    #x0 = np.zeros([n]).reshape(-1,1).T*np.nan
    
    #dQ = np.diff(Q, axis=0)
    #dQ = np.concatenate([x0, dQ])/dt

    state = [trange, q,c,Q,D,M,p,pstd,Me,Qe,en,Metot, ap, ac, ae]
    

    
    print('Done...')
    
    return state

def plot(state, *, lw1=0.5, lw2=0.1):
    
    trange,q,c,Q,D,M,p,pstd,Me,Qe,en,Metot, ap, ac, ae = state
    print('Plotting...')  
    
      
    ###
    plt.close()
    lw = lw1
    
    plt.rcParams['figure.dpi'] = 500
    
    f, axs = plt.subplots(2)
    axs[0].plot(trange, q[:,0], lw=lw, c='b')
    axs[1].plot(trange, q[:,1], lw=lw, c='b')
    axs[0].plot(trange, c[:,0], lw=lw, ls='-', c='r')
    axs[1].plot(trange, c[:,1], lw=lw, ls='-', c='r')
    axs[0].set_ylabel('q1,c1')
    axs[1].set_ylabel('q2,c2')
    
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
    
    lw = lw2
    log = True
    
    f, axs = plt.subplots(5, figsize=(8, 3*5))
    axs[0].plot(trange, en, lw=lw)
    axs[0].set_ylabel('Attempts')
    if log:
        axs[0].set_yscale('log')
    
    axs[1].plot(trange, M, lw=lw, c='k')
    axs[1].plot(trange, Metot, lw=lw)
    axs[1].set_ylabel('$\Sigma |\Delta M|$')
    if log:
        axs[1].set_yscale('log')
    
    for i in range(n):
        axs[2].plot(trange, Me[:,i], lw=lw)
    axs[2].set_ylabel('$\Sigma |\Delta M_{i}|$')
    if log:
        axs[2].set_yscale('log')
        
    for i in range(n):
        axs[3].plot(trange, Qe[:,i], lw=lw)
    axs[3].set_ylabel('$\Sigma |\Delta Q_{i}|$')
    if log:
        axs[3].set_yscale('log')
    
    for i in range(n):
        axs[4].plot(trange, p[:,i], lw=lw)
        axs[4].scatter(trange, p[:,i], lw=lw)
        
    axs[4].set_ylabel('$p_{i}$')
    if log:
        axs[4].set_yscale('log')
        
def randomQ(Q):
    
    Q_ = np.zeros(n)
    
    if np.random.rand() < 0.1:
        
        i = np.random.randint(n)
        
        Q_[i] = Q[i]
        
    return Q_




T = 1
dt = 1
plt.close('all')


n = 2
Q = np.ones(n)*10
#Q[1] = Q[1]
D = np.ones(n)*3
M = 100
cg = np.ones(n)/3.
cg[1] = cg[1]*2
xi = [0, 0]# [3, 6]

N = 10
n_actions = 3
p_perturb = 0

verbose = True
alpha = 0.1# 0.01 #0.1# 0.1
beta = 0.05 #0.02 #0.15
gamma = 0.8
epsilon = 0.0#0.01#0.001#0.005
max_agent_tries = 1
max_tries = N*max_agent_tries


agent_list = [Agents.Agent(randomQ(Q),D,M,cg, n_actions=n_actions, input_dims=[2*n]) for agents in range(N)]

# Specify MOP
ProductDict = {}
for agent in agent_list:
    ProductDict[agent] = np.random.randint(0, n)
    
    agent.p_buy = np.random.uniform(M, size=n)
    agent.p_sell = np.random.uniform(M, size=n)
    
ProductDict[agent_list[0]] = 0
ProductDict[agent_list[1]] = 1

agent_list[0].Q = np.array([Q[0], 0])
agent_list[1].Q = np.array([0, Q[1]])

print('Mean Product Distribution')
dist = np.unique([i for i in ProductDict.values()], return_counts=True)[1]

# Distribution of products (log2 units)
ent = scipy.stats.entropy(dist, base=2)
print(ent)
print()


market = Markets.Marketv3(n, verbose=verbose, minimum_price = 0.001, alpha=alpha, beta=beta, gamma=gamma, epsilon=epsilon, max_agent_tries=max_agent_tries, max_tries=max_tries)

print(f'Market has {max_tries} max tries')

#main

state = main(agent_list, p_perturb, T, dt)
plot(state, lw1=0.5, lw2=0.5)