# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:27:17 2023

@author: Alex
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

class SimPlotting:
    
    def __init__(self, sim, *, dpi=500):
        
        self.dpi = dpi
        self.sim = sim
        
        self.update()
        
    def update(self):
        
        self.Nagents = self.sim.n_agents
        self.Nproducts = self.sim.n
        self.action_space = self.sim.env.action_space
        self.dt = self.sim.env.dt
        
        self.trange = self.sim.trange
        self.Aarray = self.sim.Aarray
        self.qarray = self.sim.qarray
        self.carray = self.sim.carray
        self.Qarray = self.sim.Qarray
        self.Darray = self.sim.Darray
        self.Rarray = self.sim.Rarray
        
        self.Marray = self.sim.Marray
        
        self.narray = self.sim.narray
        self.mMarray = self.sim.mMarray
        self.mEarray = self.sim.mEarray
        self.parray = self.sim.parray

    
    #def plotM(self, )
    def plotM(self, ax, lw1, lw2):
        
        Marray = self.Marray
        
        [ax.plot(self.trange, Marray[:,n], lw=lw1) for n in range(self.Nagents)]

        ax.set_xlabel('$t$')
        ax.set_ylabel(r'$M^{\nu}$')        

    def plotmN(self, ax, lw1, lw2):
        
        narray = self.narray
        ax.plot(self.trange, narray, lw=lw1)
        
        ax.set_xlabel('$t$')
        ax.set_ylabel('Market Attempts')
        
    def plotmM(self, ax, lw1, lw2):
        
        mMarray = self.mMarray
        
        t0 = self.dt/2.0
        
        [ax.plot(self.trange + t0, mMarray[:,i], label='rf$|\Delta M_{i}|$', lw=lw2) for i in range(self.Nproducts)]
        ax.plot(self.trange + t0, mMarray.sum(axis=1), c='grey',  label='$|\Delta M|$', lw=lw1)

        ax.set_xlabel('$t$')
        ax.set_ylabel('Money in Market')
        
    def plotmE(self, ax, lw1, lw2):
        
        mEarray = self.mEarray
        
        t0 = self.dt/2.0
        
        [ax.plot(self.trange + t0, mEarray[:,i], label='rf$|\Delta E_{i}|$', lw=lw1) for i in range(self.Nproducts)]

        ax.set_xlabel('$t$')
        ax.set_ylabel('Quantities in Market')   
        
    def plotp(self, ax, lw1, lw2):
        
        parray = self.parray
        
        t0 = self.dt/2.0
        
        [ax.plot(self.trange + t0, parray[:,i], label='rf$p_{i}$', lw=lw1) for i in range(self.Nproducts)]
        
        ax.set_xlabel('$t$')
        ax.set_ylabel('Prices')
        
    def plotMarket(self, *, mw=4, mh=8, Mlog=False, mNlog=False, mMlog=False, mElog=False, plog=False, tight_layout=True, **kwargs):
    
        fig, axs = plt.subplots(5, figsize=(mw, mh))
    
        self.plotM(axs[0], **kwargs)
        self.plotmN(axs[1], **kwargs)
        self.plotmM(axs[2], **kwargs)
        self.plotmE(axs[3], **kwargs)
        self.plotp(axs[4], **kwargs)
        
        if Mlog:
            axs[0].set_yscale('log')
        if mNlog:
            axs[1].set_yscale('log')
        if mMlog:
            axs[2].set_yscale('log')
        if mElog:
            axs[3].set_yscale('log')
        if plog:
            axs[4].set_yscale('log')
            
        if tight_layout:
            plt.tight_layout()
            
        
        
        
        
    def _plotXi(self, ax, i, Xarray, X, lw1, lw2):
        
        if (X=='q') or (X=='c') or (X=='cg'):
            t0 = self.dt/2.0
        else:
            t0 = 0.0
            
        [ax.plot(self.trange - t0, Xarray[:,i,n], lw=lw2) for n in range(self.Nagents)]
    
        Xtot = Xarray[:,i].sum(axis=1)/(float(self.Nagents))
        ax.plot(self.trange - t0, Xtot, c='k', lw=lw1)
        
    def Qi(self, i, *, lw1=0.1, lw2=0.05, w=4,h=2, xmin=None, xmax=None, ymin=None, ymax=None):
        
        f, ax = plt.subplots(figsize=(w,h))
        
        self.plotQi(ax, i, lw1=lw1, lw2=lw2, w=w, h=h, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        ax.set_xlabel('$t$')
        ax.set_ylabel(fr'$\bar{{Q}}_{i}$')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        

    def plotqi(self, ax, i, *, lw1=0.1, lw2=0.05, w=4,h=2, xmin=None, xmax=None, ymin=None, ymax=None):
        
        Xarray = self.qarray
        self._plotXi(ax, i, Xarray, 'q', lw1, lw2) 
        
    def plotci(self, ax, i, *, lw1=0.1, lw2=0.05, w=4,h=2, xmin=None, xmax=None, ymin=None, ymax=None):
        
        Xarray = self.carray
        self._plotXi(ax, i, Xarray, 'c', lw1, lw2)   
    
    def plotQi(self, ax, i, *, lw1=0.1, lw2=0.05, w=4,h=2, xmin=None, xmax=None, ymin=None, ymax=None):
        
        Xarray = self.Qarray
        self._plotXi(ax, i, Xarray, 'Q', lw1, lw2)
        
    def plotDi(self, ax, i, *, lw1=0.1, lw2=0.05, w=4, h=2, xmin=None, xmax=None, ymin=None, ymax=None):
    
        Xarray = self.Darray
        self._plotXi(ax, i, Xarray, 'D', lw1, lw2)
        
    def plotQuantities(self, *, mw=4, mh=8, qmin=None, qmax=None, cmin=None, cmax=None, Qmin=None, Qmax=None, Dmin=None, Dmax=None, qlog=False, clog=False, Qlog=True, Dlog=True, tight_layout=False, **kwargs):
    
        fig, axs = plt.subplots(4, self.Nproducts, figsize=(mw*self.Nproducts,mh), squeeze=False) 
    
    
        #kwargs = {'lw1':lw1, 'lw2':lw2}
    
        # Plot Q values for specific product i
        axs[0,0].set_ylabel(r'$\bar{q}$')
        axs[1,0].set_ylabel(r'$\bar{c}$')
        axs[2,0].set_ylabel(r'$\bar{Q}$')
        axs[3,0].set_ylabel(r'$\bar{D}$')
    
    
        for i in range(self.Nproducts):
            axs[0,i].set_title(f'Product {i+1}')
    
    
            self.plotqi(axs[0,i], i, **kwargs)
            self.plotci(axs[1,i], i, **kwargs)
            self.plotQi(axs[2,i], i, **kwargs)
            self.plotDi(axs[3,i], i, **kwargs)
    
            axs[0,i].set_ylim(qmin, qmax)
            axs[1,i].set_ylim(cmin, cmax)
            axs[2,i].set_ylim(Qmin, Qmax)
            axs[3,i].set_ylim(Dmin, Dmax)
    
            if qlog:
                axs[0,i].set_yscale('log')
            if clog:
                axs[1,i].set_yscale('log')
            if Qlog:
                axs[2,i].set_yscale('log')
            if Dlog:
                axs[3,i].set_yscale('log')
    
        if tight_layout:
            plt.tight_layout()
        

    def action_probabilities(self):
        
        X = [(self.Aarray == a).sum(axis=1) for a in self.action_space]
        X = np.vstack(X)/float(self.Nagents)
        
        return X        

    # Old. Plots all actions (doesn't combine produce actions into one 'labor action')
    def _old_plotA(self, *, lw1=0.05, lw2=0.05, w=4,h=2, xmin=None, xmax=None, ymin=None, ymax=None):
        
        Aprob = self.action_probabilities()
        
        f, ax = plt.subplots()
        [ax.plot(self.trange-self.dt/2.0, Aprob[a], lw=lw1, label=fr'$\varpi_{a}$') for a in self.action_space]
        
        ax.plot(self.trange-self.dt/2.0, self.sim.epsilon_list, c='k', lw=lw1, label='$\epsilon$')
        
        
        ax.set_xlabel('$t$')
        #ax.set_ylabel(r'$\Sigma_\nu A^{\nu}$')
        ax.set_ylabel('$N_A$')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title('Action Probabilities')
        plt.legend()
        
    def _ewm(self, X, alpha, axis=0):
        '''

        Parameters
        ----------
        X : Array
            DESCRIPTION.
        alpha : Exponential Weight Scaling
            DESCRIPTION.
        axis : 0 if mean along rows. 1 if mean along columns
            DESCRIPTION. The default is 0.

        Returns
        -------
        X

        '''
        
        return pd.DataFrame(X).ewm(alpha=alpha, axis=axis).mean().values
        
        
    def plotA(self, *, lw1=0.5, lw2=0.1, w=4,h=2, xmin=None, xmax=None, ymin=None, ymax=None, ewm=False, alpha=0.5):
        
        action_name = {0: '0', self.Nproducts+1: 'C', self.Nproducts+2: 'E'}
        action_color = {0: 'C0', self.Nproducts+1: 'C2', self.Nproducts+2: 'C3'}
        
        Aprob = self.action_probabilities()
        
        Lprob = Aprob[1:self.Nproducts+1, :].sum(axis=0)
        
        f, ax = plt.subplots()
        
        if ewm:
            
            [ax.plot(self.trange-self.dt/2.0, Aprob[a], lw=lw2, c=action_color[a]) for a in [0,self.Nproducts+1, self.Nproducts+2]]
            ax.plot(self.trange-self.dt/2.0, Lprob, lw=lw2, c='C1')
            
            #Aprob = self._ewm(Aprob, alpha)
            #Lprob = self._ewm(Lprob, alpha)
            
            [ax.plot(self.trange-self.dt/2.0, self._ewm(Aprob[a], alpha), lw=lw1, c=action_color[a], label=fr'$\varpi_{action_name[a]}$') for a in [0,self.Nproducts+1, self.Nproducts+2]]
            ax.plot(self.trange-self.dt/2.0, self._ewm(Lprob, alpha), lw=lw1, c='C1', label=r'$\varpi_L$')
        
        
        else:
            
            [ax.plot(self.trange-self.dt/2.0, Aprob[a], lw=lw1, c=action_color[a], label=fr'$\varpi_{action_name[a]}$') for a in [0,self.Nproducts+1, self.Nproducts+2]]
            ax.plot(self.trange-self.dt/2.0, Lprob, lw=lw1, c='C1', label=r'$\varpi_L$')
        
        ax.plot(self.trange-self.dt/2.0, self.sim.epsilon_list, c='k', lw=lw1, label='$\epsilon$')
        
        
        ax.set_xlabel('$t$')
        #ax.set_ylabel(r'$\Sigma_\nu A^{\nu}$')
        ax.set_ylabel('$N_A$')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title('Action Probabilities')
        plt.legend()
        
        
    def agent_probabilities(self, n):
    
        agent_A = self.Aarray[1:,n]
    
        unique_actions = self.action_space
    
        probabilities = []
    
        for action in unique_actions:
    
            action_TF = np.where(agent_A == action, 1, 0)
            running_mean = np.cumsum(action_TF)/np.arange(1, len(action_TF)+1)
    
            probabilities.append(running_mean)
    
        probabilities = np.array(probabilities)
    
        n_actions = len(unique_actions)
    
        new_column = np.zeros((n_actions, 1))*np.nan
        probabilities = np.hstack((new_column, probabilities))
    
        return probabilities 

    def agent_entropy(self, n, *, base=2):
        '''
        Calculate the agent's action entropy based off their running action-probability
        Note that this is defined for a single individual and the probabilities are running means
        So it is not the same as using the number of each specific action taken at time t.
        e.g. at time t if all agents take action 2, the agent probabilities won't be 100% for action 2 since the probabilities are running means
        So the entropy won't be zero in this case.
        '''
        
        p_n = self.agent_probabilities(n)
        H_n = stats.entropy(p_n, base=base)
    
        return H_n
            


    def plotPolicy(self, a, *, lw1=0.1, lw2=0.3, ewm=False, alpha=0.5, count=True, ymin=None, ymax=None):
    
        '''
        Plots specific action numbers (how many times it is performed) if count=True
        And underlying agent action probabilities (action probability per agent)
    
        Note, that the action numbers count the proportion of agents undergoing action a - defined system wide
        The agent action probabilities are a running probability that an agent takes action a - defined per agent
    
        a : int (sector ID) or string ('sumlabor') for plotting all labor actions as one
        lw1 : Linewidth for action counts
        lw2 : Linewidth for agent probabilities
        count : bool To include action counts or no.
        ewm : bool For using the exponentially weighted mean to show action counts
        alpha : float if ewm TRUE then the alpha used for exp. weighted mean
    
        '''
        if a=='C':
            a = self.Nproducts+1
        elif a=='E':
            a = self.Nproducts+2
            
        action_name = {0: '0', self.Nproducts+1: 'C', self.Nproducts+2: 'E', 'L': 'L'}
    
        try:
            action = action_name[a]
        except KeyError:
            action = a
    
        f, ax = plt.subplots()
    
        for n in range(self.Nagents):
    
            # Agent n probability vector for each time 
            pvec_n = self.agent_probabilities(n)
    
            if a != 'L':
                pa_n = pvec_n[a]
    
            else:
                # Combine all labor into one if asked too
                pa_n = pvec_n[1:self.Nproducts+1, :].sum(axis=0)
    
            ax.plot(self.trange - self.dt/2.0, pa_n, lw=lw2)
    
        p = self.action_probabilities()
    
        if a != 'L':
            pa = p[a]
    
        else:
            # Combine all labor into one if asked too
            pa = p[1:self.Nproducts+1, :].sum(axis=0)
    
        if count:
            if ewm:
                ax.plot(self.trange - self.dt/2.0, self._ewm(pa, alpha), lw=lw1, c='k')
            else:
                ax.plot(self.trange - self.dt/2.0, pa, lw=lw1, c='k')
    
        ax.set_xlabel('$t$')
        ax.set_ylabel(r'$\varpi^{\nu}$')
        #ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f'Action {action} Probabilities')
        #plt.legend()

        
        
        
    #def plotL(self, *, lw1, lw2)
    # Plot the actions ranging from a=1 to a=Nproducts
    # Have option to show each agent. Or EWM of each agent
        

    """    
    def _plotXi(self, Xarray, i, w, h, lw1, lw2, xmin, xmax, ymin, ymax, X):
        
        if (X=='q') or (X=='c') or (X=='cg'):
            t0 = self.dt/2.0
        else:
            t0 = 0.0
        
        f, ax = plt.subplots(figsize=(w,h))
    
        [ax.plot(self.trange - t0, Xarray[:,i,n], lw=lw2) for n in range(self.Nagents)]
    
        Xtot = Xarray[:,i].sum(axis=1)/(float(self.Nagents))
        ax.plot(self.trange - t0, Xtot, c='k', lw=lw1)
        
        ax.set_xlabel('$t$')
        ax.set_ylabel(fr'$\bar{{{X}}}_{i}$')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        return f, ax
        
    def plotQi(self, i, *, lw1=0.1, lw2=0.05, w=4,h=2, xmin=None, xmax=None, ymin=None, ymax=None):
    
        f, ax = self._plotXi(self.Qarray, i, w, h, lw1, lw2, xmin, xmax, ymin, ymax, 'Q')
    
        #return f,ax
        

    def plotDi(self, i, *, lw1=0.1, lw2=0.05, w=4,h=2, xmin=None, xmax=None, ymin=None, ymax=None):
    
        f, ax = self._plotXi(self.Darray, i, w, h, lw1, lw2, xmin, xmax, ymin, ymax, 'D')
    
        return f,ax
    
    def plotqi(self, i, *, lw1=0.1, lw2=0.05, w=4,h=2, xmin=None, xmax=None, ymin=None, ymax=None):
    
        f, ax = self._plotXi(self.Qarray, i, w, h, lw1, lw2, xmin, xmax, ymin, ymax, 'q')
    
        return f,ax
    
    def plotci(self, i, *, lw1=0.1, lw2=0.05, w=4,h=2, xmin=None, xmax=None, ymin=None, ymax=None):
    
        f, ax = self._plotXi(self.Qarray, i, w, h, lw1, lw2, xmin, xmax, ymin, ymax, 'c')
    
        return f,ax
    """
    