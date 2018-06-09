#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 08:51:32 2018

@author: jens
"""

import numpy as np
import matplotlib.pyplot as plt


mu = 2

def f(t,hist):
    a = 1
    b = 2
    return mu + (a*np.exp(-b*(t-hist))*((t-hist) > 0)).sum()

tmax = 5
z = 10

hist = np.array([])
num = tmax*z

X = np.random.rand(2, num)
X = X[:,np.argsort(X[0,:])]
X[0,:] = X[0,:]*tmax
X[1,:] = X[1,:]*z 

t_cur = 0
ind = [] # gives indices in X for events

for item in X.T:
    if item[0] > t_cur:
        t_eval = item[0]
        if len(hist) == 0:
            if mu > item[1]:
                hist = np.append(hist,t_eval)
                t_cur = t_eval
                ind.append(np.argwhere(X[0] == item[0])[0,0])
            else:
                t_cur = t_eval
        else:
            if f(t_eval, hist) > item[1]:
                hist = np.append(hist,t_eval)
                t_cur = t_eval
                ind.append(np.argwhere(X[0] == item[0])[0,0])
            else:
                t_cur = t_eval

time = np.append(np.linspace(0,tmax, 10000), hist)
time = np.sort(time)

#%% Find parent-children structure

families = [[0,0]]
SpatInt = X.take(ind, axis = 1)
SpatInt = SpatInt[:,SpatInt[1,:] > mu].T
print(SpatInt)
for item in SpatInt:
    for i in range(len(hist) - 1):
        if (f(item[0], hist[:i]) <= item[1]) and (item[1] < f(item[0], hist[:i+1])):
            parent = np.argwhere(X[0,:] == hist[i])[0,0]
            child = np.argwhere(X[1,:] == item[1])[0,0]
            families.append([parent,child])

families.remove([0,0])
print(families)

#%% Plotting time
def f_plot(t,hist):
    a = 1
    b = 2
    mu = 2
    return mu + (a*np.exp(-b*(t-hist))*((t-hist) > 0)).cumsum()

f_pl = [f(s,hist) for s in time]
f_cum_pl = [f_plot(s,hist) for s in time]

ax = plt.gca()    
plt.plot(X[0,:], X[1,:], 'xc')
plt.plot(hist, np.zeros(len(hist)), 'ro')
plt.plot(time, f_pl, 'b')
for item in families:
    plt.plot([X[0,item[0]], X[0,item[1]]], [X[1,item[0]], X[1,item[1]]], 'g:')
plt.plot(time, f_cum_pl)
plt.plot(time, f_pl, 'b')
ax.set_xlim([0,tmax])
ax.set_ylim([-0.2,z])
plt.xlabel('Zeit')
plt.ylabel('Intensität')
plt.legend(('Poisson Prozess', 'Eventzeiten', 'Intensität', 'Abstammung'), loc = 1)
plt.show()