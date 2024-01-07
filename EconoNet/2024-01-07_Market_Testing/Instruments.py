# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 21:27:02 2023

@author: Alex
"""

import numpy as np


class IoP():
    
    def __init__(self):
        pass
    
class Universal(IoP):
    
    def __init__(self, prod_vec, *, dim=None, Nmin=1, Nmax=None):
        
        self._prod_vec = prod_vec
        
        if (type(prod_vec) == list) or (type(prod_vec) == np.ndarray):
            self._matrix = np.diagflat(self._prod_vec)
            self.dim = self._matrix.shape[0]
            
        elif (type(prod_vec) == int) or (type(prod_vec) == float):
            self.dim = dim
            assert (self.dim is not None)
            diag = np.full(self.dim, prod_vec)
            self._matrix = np.diagflat(diag)
            
        
        self.Nmin = Nmin
        self.Nmax = Nmax
        
        if self.Nmax is not None:
            assert self.Nmin >= self.Nmax
            
    @property 
    def matrix(self):
        return self._matrix
    
    @property
    def prod_vec(self):
        return self._prod_vec
    
class LinearIoP(IoP):
    
    def __init__(self, prod_vec, *, sector=None, dim=None, Nmin=1, Nmax=None):
        
        self._prod_vec = prod_vec
        
        if (type(prod_vec) == list) or (type(prod_vec) == np.ndarray):
            self._matrix = np.diagflat(self.prod_vec) # use diagflat so the output is always 2d
            # If sector not explicit
            # It can be found from _matrix where nonzero
            # but if multiple sectors present, then maybe sector should be a list   
            
        elif (type(prod_vec) == int) or (type(prod_vec) == float):
            self.sector = sector
            self.dim = dim
            assert (self.sector is not None)
            assert (self.dim is not None)
            diag = np.zeros(self.dim)
            diag[self.sector] = self._prod_vec
            self._matrix = np.diagflat(diag)
            
            
        self.Nmin = Nmin
        self.Nmax = Nmax
        
        if self.Nmax is not None:
            assert self.Nmin >= self.Nmax
            
    @property 
    def matrix(self):
        return self._matrix
    
    @property
    def prod_vec(self):
        return self._prod_vec