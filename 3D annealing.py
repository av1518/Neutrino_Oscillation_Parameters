import numpy as np
import matplotlib.pyplot as plt
import project_functions as fn

plot_params = {'axes.labelsize':14,
          'axes.titlesize':14,
          'font.size':15,
          'figure.figsize':[10,8],
          'font.family': 'Times New Roman'}
plt.rcParams.update(plot_params)
plt.rcParams.update(plot_params)
#%% 
data = np.loadtxt('observed.txt') #obserced mu
sim = np.loadtxt('simulated.txt') # simulated mu - unoscillated event rate prediction

t = fn.runchain3(fn.nll3, 0.5, 2.4e-3,1.5, sim, data)
#%%
lists = []
for i in range(len(t)):
    lists.append(t[i])

anneal= np.vstack((lists))
#%%
iterationsx = np.arange(0,len(anneal),1)
nllvalsy = [fn.nll3(*i) for i in anneal]
#%%
plt.plot(iterationsx, nllvalsy)
plt.xlabel('Iteration Number')
plt.ylabel('NLL Value')
#%%
plt.plot(np.arange(0,len(anneal),1),anneal[:,0])
plt.ylabel(r'$\theta$ (rad)')
plt.xlabel(r'Iteration number')
# plt.title(r'Simulated annealing for T = 3 to 0.5 in 0.5 steps, 1e5 iterations per T')
#%%
plt.plot(np.arange(0,len(anneal),1),anneal[:,1])
plt.ylabel(r'$\Delta m^{2}$ $(eV^{2})$')
plt.xlabel(r'Iteration number')
# plt.title(r'Simulated annealing for T = 3 to 0.5 in 0.5 steps, 1e5 iterations per T')

#%%
plt.plot(np.arange(0,len(anneal),1),anneal[:,2])
plt.ylabel(r'$\alpha$ $(eV^{-1})$ ')
plt.xlabel(r'Iteration number')
# plt.title(r'Simulated annealing for T = 3 to 0.5 in 0.5 steps, 1e5 iterations per T')
#%% parameter extraction
thetabar = np.sum(anneal[300000:,0])/(len(anneal)-300000)
print(f'theta_min through annealing = {thetabar}') # seems to get stuck to the other minimum above np.pi/4, must restring range
dmbar = np.sum(anneal[300000:,1])/(len(anneal)-300000)
print(f'dm_min through annealing = {dmbar}')
abar = np.sum(anneal[300000:,2])/(len(anneal)-300000)
print(f'a_min through annealing = {abar}')
params = [thetabar,dmbar,abar]
print('nll for these values = ', fn.nll3(*params))
errorsanneal = fn.nllerror3(*params)
print(errorsanneal)
#%%

newsim = fn.oscillated_sim3(sim,*params)

bins = np.arange(0,10,0.05)

plt.subplot(2,1,1)
plt.hist(bins, bins, weights = data, label = r'Observed $\nu_{\mu}$ count') # treating each bin as a single points with a weight equal to its count
plt.legend()
plt.xlabel('Energy (GeV)')
plt.ylabel('Counts')

plt.subplot(2,1,2)
plt.hist(bins, bins, weights = newsim, label = r'Expected $\nu_{\mu}$ count') # treating each bin as a single points with a weight equal to its count
plt.legend()
plt.xlabel('Energy (GeV)')
plt.ylabel('Counts')
# plt.title('Simulated data with simulated annealing parameters')
