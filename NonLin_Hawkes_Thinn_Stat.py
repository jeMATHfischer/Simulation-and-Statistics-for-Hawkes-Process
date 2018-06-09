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
print(count_2/Simulations)
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
import matplotlib.pyplot as plt
import numpy as np
from Hawkes_Thinning_class import Hawkes


np.random.seed(42)


def deleter(Hawkes):
    k = 1
    if Hawkes.param < 5:
        Hawkes.propogate_by_amount(100)
    else: 
        Hawkes.propogate_by_amount(10)
        while (Hawkes.density(Hawkes.Events[-1]) < 100) and (k < 1000): 
            Hawkes.propogate_by_amount(10)
            k += 1
        return k


def HawkesIntensity_temporal(time, b):
    a = 1
    IndInTemp = (time < b/2) & (time > 0)
    IndDecTemp = (time >= b/2) & (time < b)
    return 2*a/b * (time)*IndInTemp + ((-(2*a)/b)* (time) + 2*a )*IndDecTemp

expl_proc = []
param = 3

count_1 = 0

for i in range(100):
    H = Hawkes(HawkesIntensity_temporal, param, phi = lambda s: (1+s)**2/(2+s), mon_kernel = False)
    deleter(H)
    if (H.density(H.Events[-1]) >= 100) or (H.param < 5): #1/Area # or (H.param < q) for nonexpl process, q appropriately    
        vars()['H_' + str(count_1)] = H
        expl_proc.append('H_' + str(count_1))
        count_1 += 1

print(count_1)

for item in expl_proc:
    vars()['dense_' + item] = []
    t = np.linspace(0,vars()[item].Events[-1],1000)
    for i in t:
        vars()['dense_' + item].append(vars()[item].density(i))
    plt.plot(t,vars()['dense_' + item])

plt.xlabel('Zeit')
plt.ylabel('Ereignis-Dichte')
plt.show()
#
#count_2 = 0
#bars = []
#final = []
#Simulations = 10
#
#for i in range(Simulations):
#    H = Hawkes(HawkesIntensity_temporal, param, mon_kernel = False)
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
import matplotlib.pyplot as plt
import numpy as np
from Hawkes_Thinning_class import Hawkes


np.random.seed(42)


def deleter(Hawkes):
    k = 1
#    if Hawkes.param < 2:
#        Hawkes.propogate_by_amount(100)
#    else: for nonexploding process with Hawkes.param < q, q approprietly
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


def counting(param):
    count_1 = 0
    for i in range(100):
        H = Hawkes(HawkesIntensity_temporal, param, phi = lambda s: (1+s)**2/(2+s), mon_kernel = False) #np.exp(s), 3*np.log(2+s)
        deleter(H)
        if (H.density(H.Events[-1]) >= 100): #1/Area # or (H.param < q) for nonexpl process, q appropriately    
            count_1 += 1
        
    print(count_1)
    return count_1

params = np.linspace(0.1,1.9,10)


plt.step(params, [counting(param) for param in params])
plt.xlabel('Parameter-Werte')
plt.ylabel('Anzahl Trajektorien')
plt.show()

