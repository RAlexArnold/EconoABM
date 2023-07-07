# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 00:56:25 2023

@author: Alex
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_states(trange, Aarray, qarray, Carray, Qarray, Darray, *, lw=0.5):
    
    dt = np.diff(trange).mean()
    tmax = trange.max()
    n = qarray.shape[1]
    
    n_actions = np.nanmax(Aarray)

    plt.rcParams['figure.dpi'] = 500
    f, ax = plt.subplots(nrows=5, figsize=(8,5*2))
    
    ax[0].plot(trange-dt/2.0, Aarray, lw=lw)
    ax[0].set_ylim(-0.1,n_actions+0.1)
    ax[0].set_title('Actions')
    
    [ax[1].plot(trange-dt/2.0, qarray[:,i], label=f'q{i}', lw=lw) for i in range(n)]
    #[ax[1].scatter(trange-dt/2.0, Carray[:,i]) for i in range(n)]
    ax[1].set_title('Production')
    ax[1].legend()
    
    
    [ax[2].plot(trange-dt/2.0, Carray[:,i], label=f'C{i}', lw=lw) for i in range(n)]
    #[ax[1].scatter(trange-dt/2.0, Carray[:,i]) for i in range(n)]
    ax[2].set_title('Consumption')
    ax[2].legend()
    
    
    [ax[3].plot(trange, Qarray[:,i], label=f'Q{i}', lw=lw) for i in range(n)]
    #[ax[3].scatter(trange, Qarray[:,i]) for i in range(n)]
    ax[3].set_title('Q Stock')
    ax[3].legend()
    
    [ax[4].plot(trange, Darray[:,i], label=f'D{i}', lw=lw) for i in range(n)]
    #[ax[4].scatter(trange, Darray[:,i]) for i in range(n)]
    ax[4].set_title('Consumption Defecit')
    ax[4].legend()
    
    [ax[i].set_xlim(-dt/2.0,tmax) for i in range(5)]
    
    plt.tight_layout()
    
def plot_decisions(trange, Darray, Rarray, epsilon_list, Parray, omega, *, lw=0.5):
    
    n_actions = Parray.shape[0]
    
    dt = np.diff(trange).mean()
    tmax = trange.max()
    n = Darray.shape[1]
    
    nrows = 3
    f, ax = plt.subplots(nrows=nrows, figsize=(8,nrows*2))
    
    [ax[0].plot(trange, Darray[:,i], label=f'D{i}', lw=lw) for i in range(n)]
    ax[0].set_title('Consumption Defecit')
    #ax[0].set_yscale('log')
    ax[0].legend()
    
    ax[1].plot(trange, Rarray, lw=lw)
    #ax[1].plot(trange, RavgArray, c='k', lw=0.5)
    ax[1].set_title('Reward')
    #ax[1].set_yscale('log')
    
    #[ax[2].plot(trange, Harray[:,i], label=f'$\\theta_{i}$') for i in range(n)]
    #[ax[2].scatter(trange, Harray[:,i]) for i in range(n)]
    #ax[2].set_title('Preferences')
    #ax[2].legend()
    
    [ax[2].plot(trange, Parray[i],  label=f'$\\varpi_{i}$', lw=lw) for i in range(n_actions)]
    #[ax[2].scatter(trange, Parray[i]) for i in range(n)]
    ax[2].plot(trange, epsilon_list, label='$\epsilon$', lw=lw, c='k')
    ax[2].axhline(omega, lw=lw, c='k', ls='--')
    ax[2].set_title('Probabilities')
    ax[2].legend()
    
    [ax[i].set_xlim(-dt/2.0,tmax) for i in range(nrows)]
    #[ax[i].set_yscale('log') for i in range(nrows)]
    
    plt.tight_layout()