import numpy as np
import matplotlib.pyplot as plt

def f(t,hist):
    a = 1
    b = 2
    mu = 2
    return mu + (a*np.exp(-b*(t-hist))*((t-hist) > 0)).sum()

tmax = 10 
z = 12

hist = np.array([])
num = tmax*z

X = np.random.rand(2, num)
X = X[:,np.argsort(X[0,:])]
X[0,:] = X[0,:]*tmax
X[1,:] = X[1,:]*z 

t_cur = 0

for item in X.T:
    if item[0] > t_cur:
        t_eval = item[0]
        if len(hist) == 0:
            if f(t_eval, 0) > item[1]:
                hist = np.append(hist,t_eval)
                t_cur = t_eval
            else:
                t_cur = t_eval
        else:
            if f(t_eval, hist) > item[1]:
                hist = np.append(hist,t_eval)
                t_cur = t_eval
            else:
                t_cur = t_eval

time = np.append(np.linspace(0,tmax, 10000), hist)
time = np.sort(time)

def f_plot(t,hist):
    a = 2
    b = 4
    mu = 2
    return mu + (a*np.exp(-b*(t-hist))*((t-hist) > 0)).cumsum()

ind = [f_plot(s,hist) for s in time]

ax = plt.gca()
plt.plot(X[0,:], X[1,:], 'xc')
plt.plot(hist, np.zeros(len(hist)), 'ro')
plt.plot(time, ind)
ax.set_xlim([0,tmax])
ax.set_ylim([-0.2,z])
plt.xlabel('Zeit')
plt.ylabel('Intensität')
plt.legend(('Poisson Prozess', 'Eventzeiten', 'Intensität'), loc = 1)
plt.show()