#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:59:18 2018

@author: jens
"""


import matplotlib.pyplot as plt
import numpy as np
from Hawkes_Thinning_class import Hawkes


np.random.seed(42)


def deleter(Hawkes, alpha, th):
    k = 1
    if alpha*Hawkes.param[0]*Hawkes.param[1]/2 < 1:
        Hawkes.propogate_by_amount(10)
        while Hawkes.Events[-1] < 500:
            Hawkes.propogate_by_amount(1)
    else: 
        Hawkes.propogate_by_amount(10)
        while (Hawkes.density(Hawkes.Events[-1]) < th) and (k < 1000): 
            Hawkes.propogate_by_amount(10)
            k += 1
        return k
#
#def HawkesIntensity_temporal(time, params):
#    IndInTemp = (time < params[1]/2) & (time > 0)
#    IndDecTemp = (time >= params[1]/2) & (time < params[1])
#    return 2*params[0]/params[1] * (time)*IndInTemp + ((-(2*params[0])/params[1])* (time) + 2*params[0] )*IndDecTemp
#

def HawkesIntensity_temporal(time, params):
    Ind = (time >= 0)
    return params[0]*np.exp(-params[1]*time)*Ind

# alpha gives boundry of stability given by theorems by bremaud & masoulie
alpha = 1
th = 10


expl_proc = []


def counting(param, alpha):
    count_1 = 0
    for i in range(100):
        H = Hawkes(HawkesIntensity_temporal, param, phi = lambda s: (1+s)**2/(2+s), mon_kernel = False) #np.exp(s), 3*np.log(2+s)
        deleter(H, alpha, th)
        if (H.density(H.Events[-1]) >= 100): #1/Area # or (H.param < q) for nonexpl process, q appropriately    
            count_1 += 1
        
    print(count_1)
    return count_1

params_a = np.linspace(0.1,1.9,10)
params_b = np.linspace(0.1,1.9,10)

# plotly surface plot, work on it
import itertools

z = np.array([item[0]**2 + item[1]**2 for item in itertools.product(params_a,params_b)])
z = np.reshape(z, (len(params_a),-1))    