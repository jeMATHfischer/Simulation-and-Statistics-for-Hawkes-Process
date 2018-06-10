
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
param = [2,1]
alpha = 1
th = 100


expl_proc = []
count_1 = 0

for i in range(100):
    print(count_1)
    H = Hawkes(HawkesIntensity_temporal, param, mon_kernel = True)
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

#plt.plot(t, 2*np.ones(len(t)), 'k')
plt.xlabel('Zeit')
plt.ylabel('Ereignis-Dichte')
#plt.legend(tuple(expl_proc))
plt.show()
#
#count_2 = 0
#bars = []
#final = []
#Simulations = 10000
#
#for i in range(Simulations):
#    H = Hawkes(HawkesIntensity_temporal, param, phi = lambda s: np.exp(s), mon_kernel = False)
#    stop_index = deleter(H)
#    if stop_index < 100:
#        count_2 += 1
#        print('density explodes around {:08.6f}'.format(H.Events[-1]))
#        bars.append(stop_index)
#        final.append(H.Events[-1]) 
#
#plt.figure()
#plt.scatter(final, bars, c = final, cmap = 'rainbow')
#plt.xlabel('Event-Zeit vor Dichte = 1000')
#plt.ylabel('Anzahl Events bevor Dichte = 1000')
#plt.show()
#print(count_2/Simulations)
#%%
#
#count = 0
#
#while (len(expl_proc) > 20) and (count < 5):
#    for item in expl_proc:
#        vars()[item].propogate_by_amount(1000)
#        if vars()[item].Events[-1] >= 2:
#            expl_proc.remove(item)
#    
#    count += 1
#    len_tracker.append([len(expl_proc)])
#    print(len(expl_proc))
#    print(count)
#    
#plt.step(np.arange(len(len_tracker)), len_tracker)
#
#    
#%%

#%%
import matplotlib.pyplot as plt
import numpy as np
from Hawkes_Thinning_class import Hawkes


np.random.seed(42)


def deleter(Hawkes):
    k = 1
    Hawkes.propogate_by_amount(10)
    while (Hawkes.density(Hawkes.Events[-1]) < 100) and (k < 100): 
        Hawkes.propogate_by_amount(10)
        k += 1
    return k


def HawkesIntensity_temporal(time, b):
    a = 1
    IndInTemp = (time < b/2) & (time > 0)
    IndDecTemp = (time >= b/2) & (time < b)
    return 2*a/b * (time)*IndInTemp + ((-(2*a)/b)* (time) + 2*a )*IndDecTemp

expl_proc = []
param = 1/2

count_1 = 0

for i in range(100):
    H = Hawkes(HawkesIntensity_temporal, param, phi = lambda s: np.exp(s), mon_kernel = False)
    H.propogate_by_amount(100)
    if H.density(H.Events[-1]) > 100: #1/Area    
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
#plt.legend(tuple(expl_proc))
plt.show()

count_2 = 0
bars = []
final = []
Simulations = 10000

for i in range(Simulations):
    H = Hawkes(HawkesIntensity_temporal, param, phi = lambda s: np.exp(s), mon_kernel = False)
    stop_index = deleter(H)
    if stop_index < 100:
        count_2 += 1
        print('density explodes around {:08.6f}'.format(H.Events[-1]))
        bars.append(stop_index)
        final.append(H.Events[-1]) 

plt.figure()
plt.scatter(final, bars, c = final, cmap = 'rainbow')
plt.xlabel('Event-Zeit vor Dichte = 1000')
plt.ylabel('Anzahl Events bevor Dichte = 1000')
plt.show()
#print(count_2/Simulations)
