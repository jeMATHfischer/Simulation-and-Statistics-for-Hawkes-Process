#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 08:57:54 2018

@author: jens
"""

#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def sorting(X):
    indices = np.argsort(X[0,:])
    return X[:,indices]
    

def HawkesIntensity(time, space, history, a, b, mu, cube_width):
    IndInTemp = ((time - history[0,:]) < b[0]/2) & ((time - history[0,:]) > 0)
    IndDecTemp = ((time - history[0,:]) >= b[0]/2) & ((time - history[0,:]) < b[0])
    
    temporalHist = 2*a[0]/b[0] * (time - history[0,:])*IndInTemp + ((-(2*a[0])/b[0])* (time - history[0,:]) + 2*a[0] )*IndDecTemp
 #   temporal = np.sum(temporalHist)
    
    mod1 = np.array((space - history[1,:])%cube_width)
    mod2 = np.array((space - history[1,:])%cube_width)
    mod2[(space - history[1,:]) < 0] = mod2[(space - history[1,:]) < 0] - cube_width
    IndDecSpat = (mod1 < b[1]/2) & (mod1 > 0)
    IndInSpat_pos = (mod2 >= -b[1]/2) & (mod2 < 0)
    IndInSpat_neg = (mod2 - cube_width > -b[1]/2) & (mod2 - cube_width < 0)


    spatialHist = (2*a[1]/b[1] * (mod2-cube_width) + a[1]) * IndInSpat_neg+(2*a[1]/b[1] * (mod2) + a[1]) * IndInSpat_pos+ (a[1] + (-(2*a[1])/b[1])* mod1)*IndDecSpat
#    spatial = np.sum(spatialHist)
    return mu + np.multiply(spatialHist,temporalHist).sum()

#np.sum(np.sqrt(np.abs(np.multiply(spatialHist,temporalHist)-1)))

# -------------- Script
    
# -------------- parameters with values
dimension = 3    


a = np.array([1, 1]) # a[1] temporal kernel height; a[2] spatial kernel height
b = np.array([2, 1]) # b[1] temporal kernel width; b[2] spatial kernel width
mu = 0.5


tmax, cube_width, cube_height = np.array([35, 2*np.pi, 200])
space_steps, time_steps = np.array([101, 101]) 


#---------------- dummy parameters
history = np.array([[0],[0]])


# --------------- Generate uniform distributed points in cube tmax*cube_width*cube_height

#np.random.seed(42)
l = cube_width * cube_height * tmax 
amount = np.random.poisson(l)
print(l)
cube = np.multiply(np.random.rand(dimension, amount), np.transpose(np.array([[tmax, cube_width, cube_height]])
))

cube = sorting(cube)
# ------------- accept and reject points in cube according to intensity function
for time, space in cube[0:2,:].T:
    if HawkesIntensity(time, space, history, a, b, mu, cube_width) >= cube[-1,cube[0,:] == time] :#adjust args for intensity
        
        history = np.append(history,np.array([[time],[space]]), axis = 1 )
           
        
        
del cube

# ------------- plotting time

time_boundry = np.append(np.linspace(0,tmax,time_steps), history[0,:], axis = 0)
space_boundry = np.append(np.linspace(0,cube_width,space_steps), history[-1,:], axis = 0)


for boundry in [time_boundry, space_boundry]: 
    boundry.sort()    
    
print(len(history[1,:]))


plt.scatter(history[0,1:],history[-1,1:])
plt.show()  

# =============================================================================
# Z = np.empty([len(time_boundry),len(space_boundry)])
# 
# for i in range(len(time_boundry)):
#     for j in range(len(space_boundry)):
#         Z[i,j] = HawkesIntensity(time_boundry[i],space_boundry[j],history, a, b, mu, cube_width)
# 
# time, space = np.meshgrid(time_boundry,space_boundry)
# # Plot the surface.
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# 
# surf = ax.plot_surface(time, space, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# ax.set_xlabel('time')
# ax.set_ylabel('space')
# 
# plt.show()
# 
# 
# =============================================================================

#fig = plt.figure()
#ax = fig.gca(projection='3d')

#Z = HawkesIntensity(time, space, history, a, b, mu, cube_width)



#surf = ax.plot_surface(time, space, Z, cmap=cm.coolwarm,
                     #  linewidth=0, antialiased=False)
#plt.show()
 