import numpy as np
import matplotlib.pyplot as plt
import project_functions as fn

plot_params = {'axes.labelsize':14,
          'axes.titlesize':14,
          'font.size':15,
          'figure.figsize':[10,4],
          'font.family': 'Times New Roman'}
plt.rcParams.update(plot_params)
plt.rcParams.update(plot_params)

#%%
data = np.loadtxt('observed.txt') #obserced mu
sim = np.loadtxt('simulated.txt') # simulated mu - unoscillated event rate prediction

t = fn.runchain(fn.nll, 0.5, 2.4e-3, sim, data)

lists = []
for i in range(len(t)):
    lists.append(t[i])

anneal= np.vstack((lists))
#%%
plt.plot(np.arange(0,len(anneal),1),anneal[:,0])
plt.ylabel(r'$\theta$')
plt.xlabel(r'Iteration number')
# plt.title(r'Simulated annealing for T = 0 to 5 in 0.5 steps')
#%%
plt.plot(np.arange(0,len(anneal),1),anneal[:,1])
plt.ylabel(r'$\Delta m^{2}$')
plt.xlabel(r'Iteration number')
# plt.title(r'Simulated annealing for T = 0 to 5 in 0.5 steps, 1e5 iterations per T')
#%% parameter extraction
thetabar = np.sum(anneal[300000:,0])/(len(anneal)-300000)
print(f'theta_min through annealing = {thetabar}') # seems to get stuck to the other minimum above np.pi/4, must restring range
dmbar = np.sum(anneal[300000:,1])/(len(anneal)-300000)
print(f'dm_min through annealing = {dmbar}')

params = [thetabar,dmbar]
print('nll for these values = ', fn.nll(*params))
'''
Saved values for annealing of  Simulated annealing for T = 0 to 5 in 0.5 steps, 1e5 iterations per T 
= [0.7849673032052427, 0.0023928111460407666 ]
nll for these values =  373.2193715028189
'''

'''
Saved values for annealing of  Simulated annealing for T = 3 to 0.1 in 5 steps, 1e5 iterations per T 
= [0.7854766920326939, 0.002390671871475111]
nll for these values =  373.07465818939403
'''
#%% Errors
thetastd = np.std(anneal[300000:,0])
dmstd = np.std(anneal[300000:,1])
print(f'Theta std = {thetastd}, dmstd = {dmstd}')

