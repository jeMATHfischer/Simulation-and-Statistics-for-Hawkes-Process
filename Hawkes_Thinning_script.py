import matplotlib.pyplot as plt
import numpy as np
from Hawkes_Thinning_class import Hawkes


def HawkesIntensity_temporal(time):
    a = 1 
    b = 1
    IndInTemp = (time < b/2) & (time > 0)
    IndDecTemp = (time >= b/2) & (time < b)
    return 2*a/b * (time)*IndInTemp + ((-(2*a)/b)* (time) + 2*a )*IndDecTemp

def f(t):
    a = 1
    b = 2
    return (a*np.exp(-b*(t))*(t > 0))

H = Hawkes(f, mon_kernel=True)

H.propogate_by_amount(50);

x = np.linspace(0,H.Events[-1],2000)
y, z = H.current_intensity(x)
fig = plt.figure
plt.plot(y,z)
plt.plot(H.Events, np.zeros(len(H.Events)), 'ro')
plt.xlabel('Zeit')
plt.ylabel('Intensität')
plt.legend(('Intensität', 'Eventzeiten'))

plt.show()