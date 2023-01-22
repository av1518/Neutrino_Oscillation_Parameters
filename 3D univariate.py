import numpy as np
import matplotlib.pyplot as plt
import project_functions as fn

plot_params = {'axes.labelsize':12,
          'axes.titlesize':12,
          'font.size':12,
          'figure.figsize':[10,4]}
plt.rcParams.update(plot_params)
plt.rcParams.update(plot_params)

data = np.loadtxt('observed.txt') #obserced mu
sim = np.loadtxt('simulated.txt') # simulated mu - unoscillated event rate prediction
#%%
dms = 0.00253638
thetas = 0.7674447 
a = np.linspace(0.1, 2, 1000)

dms =  np.array([dms] * len(a))
thetas = np.array([thetas] * len(a))


nlla = [fn.nll3(thetas[i], dms[i], a[i]) for i in range(len(a))]    

plt.plot(a, nlla,'-', label = r'fixed $\Delta m^2$ =  %s, fixed $\theta$ = %s' % (dms[0], thetas[0]))
plt.title(r'NLL($\alpha$)')
plt.ylabel(r'NLL($\alpha$)')
plt.xlabel(r'$\alpha$')
plt.legend()


print('Approximate position of minimum(by eye) =', 0.78)
#%%
dm = 0.00253638
theta = 0.7674447 
a = 1
u = [dm,theta,a]
clock,count = fn.univariate3(theta, dm, a, sim, data)
#%%
print(f'Clockwise minimised parameters = {clock}')
likelihood = fn.nll3(*clock)
print(f'clock likelihood = {likelihood}')
unierror = fn.nllerror3(*clock, thetafactor =0.04)
print(f'errors = {unierror}')
#%% Anticlockwise
dm = 0.00253638
theta = 0.7674447 
a = 1
u = [dm,theta,a]
anticlock,count = fn.univariate3(theta, dm, a, sim, data, first = 1, threshold = 1e-2)
#%%
print(f'Clockwise minimised parameters = {anticlock}')
likelihood = fn.nll3(*anticlock)
print(f'clock likelihood = {likelihood}')
