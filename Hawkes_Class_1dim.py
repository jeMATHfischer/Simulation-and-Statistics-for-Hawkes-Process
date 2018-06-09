#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:33:48 2018

@author: jens

Simulate a spatio-temporal Hawkes process on [0,2pi]x[0,infty) with periodic bdry conditions in space.
"""

import numpy as np
import random as rand
#import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.optimize as opt

#
#rand.seed(42)

class HawkesProcess():

    def __init__(self, Base, kernel):
        self.Base = Base
        self.kernel = kernel
        self.Events = np.array([0])
        self.PoissEvent = np.array([])
        self.Sim_num = 0
        
        
    def propogate_by_amount(self, k):        
        
        for i in range(k):
            self.PoissEvent = np.append(self.PoissEvent,rand.expovariate(1))
        
        PoissProcess = np.cumsum(self.PoissEvent)
        
        for time in PoissProcess[self.Sim_num:]:   
            distorted_kernel = lambda x: np.array([self.kernel(x-j) for j in self.Events])
            
            def righthandside(t):
                dydt = self.Base + distorted_kernel(t).sum()
                return dydt
            
            I = lambda x: quad(righthandside, 0, x)[0]  - time 
            self.Events = np.append(self.Events, opt.fsolve(I, self.Events[-1]))
        
        self.Sim_num += k
        
    def current_intensity(self, x):
        y = np.sort(np.append(x,self.Events))
        return y, [np.array([self.kernel(k-j) for j in self.Events]) for k in y]
    
    
    