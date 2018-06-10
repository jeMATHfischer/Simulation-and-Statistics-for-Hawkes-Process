#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:42:32 2018

@author: jens
"""

import matplotlib.pyplot as plt
import numpy as np
from Hawkes_Thinning_class import Hawkes


np.random.seed(42)


def deleter(Hawkes, alpha, th):
    k = 1
    if (alpha*Hawkes.param[0]/Hawkes.param[1] < 1) and (not Hawkes.Nonlin(1) == np.exp(1)):
        Hawkes.propogate_by_amount(10)
        while Hawkes.Events[-1] < 500:
            Hawkes.propogate_by_amount(1)
    else: 
        Hawkes.propogate_by_amount(10)
        while (Hawkes.density(Hawkes.Events[-1]) < th) and (k < 100): 
            Hawkes.propogate_by_amount(10)
            k += 1
        return k

#def HawkesIntensity_temporal(time, params):
#    IndInTemp = (time < params[1]/2) & (time > 0)
#    IndDecTemp = (time >= params[1]/2) & (time < params[1])
#    return 2*params[0]/params[1] * (time)*IndInTemp + ((-(2*params[0])/params[1])* (time) + 2*params[0] )*IndDecTemp

def HawkesIntensity_temporal(time, params):
    Ind = (time >= 0)
    return params[0]*np.exp(-params[1]*time)*Ind



# alpha gives boundry of stability given by theorems by bremaud & masoulie
param = [1/4,5]
alpha = 1/2
th = 100


expl_proc = []
count_1 = 0

for i in range(1000):
    print(count_1)
    H = Hawkes(HawkesIntensity_temporal, param, phi = lambda s: np.exp(s), mon_kernel = True)
    deleter(H, alpha, th)
    if (H.density(H.Events[-1]) >= th) or (alpha*H.param[0]/H.param[1] < 1):     
        vars()['H_' + str(count_1)] = H
        expl_proc.append('H_' + str(count_1))
        count_1 += 1



for item in expl_proc:
    vars()['dense_' + item] = []
    t = np.linspace(0,vars()[item].Events[-1],1000)
    for i in t:
        vars()['dense_' + item].append(vars()[item].density(i))
    plt.plot(t,vars()['dense_' + item])

plt.xlabel('Zeit')
plt.ylabel('Ereignis-Dichte')
plt.show()


mini = 10000

for item in expl_proc:
    if mini == 10000:
        mini =  vars()[item].Events[-1]
    else:
        mini = min(mini, vars()[item].Events[-1])

for item in expl_proc:
    vars()['dense_' + item] = []
    t = np.linspace(0,mini,1000)
    for i in t:
        vars()['dense_' + item].append(vars()[item].density(i))    
    
mean_builder = np.array([-1])

for item in expl_proc:
    if mean_builder[0] == -1:
        mean_builder = np.array(vars()['dense_' + item]) 
    else:
        mean_builder += np.array(vars()['dense_' + item]) 

mean_builder = mean_builder/len(expl_proc)

plt.figure()
plt.plot(t, mean_builder)
plt.xlabel('Zeit')
plt.ylabel('Mittlere Ereignis-Dichte')
plt.show()
