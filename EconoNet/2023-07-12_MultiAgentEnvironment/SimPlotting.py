# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:27:17 2023

@author: Alex
"""

import matplotlib.pyplot as plt
import numpy as np

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
        
    def action_probabilities(self):
        
        X = [(self.Aarray == a).sum(axis=1) for a in self.action_space]
        X = np.vstack(X)/float(self.Nagents)
        
        return X
    
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
        

    
    def plotA(self, *, lw1=0.05, lw2=0.05, w=4,h=2, xmin=None, xmax=None, ymin=None, ymax=None):
        
        Aprob = self.action_probabilities()
        
        f, ax = plt.subplots()
        [ax.plot(self.trange-self.dt/2.0, Aprob[a], lw=lw1, label=fr'$\varpi_{a}$') for a in self.action_space]
        
        ax.set_xlabel('$t$')
        #ax.set_ylabel(r'$\Sigma_\nu A^{\nu}$')
        ax.set_ylabel('$N_A$')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title('Action Probabilities')
        plt.legend()
        
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
    