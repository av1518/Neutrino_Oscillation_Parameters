import numpy as np
import matplotlib.pyplot as plt
import project_functions as fn

plot_params = {'axes.labelsize':14,
          'axes.titlesize':12,
          'font.size':12,
          'figure.figsize':[10,8]}
plt.rcParams.update(plot_params)
plt.rcParams.update(plot_params)

data = np.loadtxt('observed.txt') #obserced mu
sim = np.loadtxt('simulated.txt') # simulated mu - unoscillated event rate prediction




dm = 2.4e-3
thetas = np.linspace(0, np.pi/4, 1000) #0 to pi/4 covers the whole range as it ~sin**(2theta) so bracket theta goes from 0 to pi/2
dms = np.array([dm] * len(thetas))
u = np.stack((thetas,dms))

nlltheta = [fn.NLL(u[:,i], sim, data) for i in range(len(thetas))]    


def NLL_theta(theta, dm = 2.4e-3): #defining NLL just for theta varying to use in minimiser
    # dm = 2.4e-3
    u = [theta,dm]
    nllvalue = fn.NLL(u,sim,data)
    return nllvalue


theta_min ,count,dx,x_latest,y_latest = fn.parabmin(NLL_theta, 0.6, np.pi/4, 1e-7)
print(r'Value of theta_min from parabolic optimisation = %a, \
      Number of iterations = %a  \
          last change in x = %a  \
              Last 3 x values in search = %a \
                  last 3 y values in search = %a'
          % (theta_min, count, dx, x_latest,y_latest)) 



def nllroot(theta):
    b = NLL_theta(theta_min) + 0.5 
    return NLL_theta(theta) - b


leftroot, leftcount = fn.bisection(nllroot, 0.710, theta_min, tol = 1e-9)
print('leftroot from bisection =', leftroot, leftcount)
leftdiff = abs(leftroot - theta_min)
print('Difference of NLL(thetaroot) - NLL(theta_min) =', NLL_theta(leftroot) - NLL_theta(theta_min))

rightroot, rightcount = fn.bisection(nllroot, theta_min,np.pi/4, tol = 1e-9)
print('rightroot from bisection =', rightroot, rightcount)
rightdiff = abs(rightroot - theta_min)
print('Difference of NLL(thetaroot) - NLL(theta_min) =', NLL_theta(rightroot) - NLL_theta(theta_min))

std = (leftdiff + rightdiff)/2
print(f'standard div = {std}')
print(f'Theta+ = {rightdiff}, theta- = {leftdiff}')


plt.subplot(2,1,1)
plt.plot(thetas, nlltheta,'-', label = r'fixed $\Delta m^2_{23} $ =  %s $eV^2$' % dms[0], color = 'blue')
plt.axvline(x = theta_min, linestyle = '-.',label = r'$\theta_{23min}$ from parabolic search',  color = 'black')
# plt.title(r'NLL($\theta_{23}$)')
plt.ylabel(r'NLL($\theta_{23}$)')
plt.xlabel(r'$\theta_{23}$ (rad)')
plt.legend()



plt.subplot(2,1,2)
plt.plot(thetas, nlltheta,'-', label = r'fixed $\Delta m^2_{23}$ =  %s $eV^2$ ' % dm, color = 'blue')
plt.axvline(x = theta_min, linestyle = '-.',label = r'$\theta_{23min}$', color = 'black' )
# plt.title(r'NLL($\theta_{23}$)')
plt.ylabel(r'NLL($\theta_{23}$)')
plt.xlabel(r'$\theta_{23} (rad)$')
plt.xlim(leftroot-0.001, np.pi/4)
plt.ylim(NLL_theta(theta_min)-0.25/4,NLL_theta(leftroot)+0.25/4)
plt.axvline(x=leftroot, label = r'$\theta^{-}_{23}$', color = 'firebrick')
plt.axvline(x = rightroot, label = r'$\theta^{+}_{23}$', color = 'firebrick')
plt.legend()

plt.show()
#%%
#%% Testing minimize_dm
theta = np.pi/4
dm = 2.4e-3
u_in = [theta,dm]

# dm_test = fn.minimize_dm(*u_in, sim, data)
# print(dm_test)
# # print(dm_min)

#%% test univariate
testing_theta,count, errors_theta = fn.univariate(*u_in, sim, data,threshold = 1e-7)
print(f'Optimised parameters by univariate minimisation of theta first = {testing_theta}')
# print(f'list of each iterations error: {errors_theta}')
#%%
testing_dm,count, errors_dm = fn.univariate(*u_in, sim, data, first = 1, threshold = 1e-7)
print(f'Optimised parameters by univariate minimisation of dm first = {testing_dm}')
# print(f'list of each iterations error: {errors_dm}')
#%%
u = [0.7677570033447106, 0.002388736046007897]
u_dmfirst = testing_dm
u_thetafirst = testing_theta
print('univariate NLL dmfirst = ',fn.NLL(u_dmfirst,sim,data))
print('univariate NLL thetafirst= ', fn.NLL(u_thetafirst,sim,data))

errors = fn.nllerror(u_thetafirst[0],u_thetafirst[1])
print(f'univariate errors (theta_+,theta_-,dm_+,dm_-) = {errors}')

#%%
dm = 3.5e-3
thetas = np.linspace(np.pi/8, 3*np.pi/8, 1000) #0 to pi/4 covers the whole range as it ~sin**(2theta) so bracket theta goes from 0 to pi/2
dm = np.array([dm] * len(thetas))
u = np.stack((thetas,dm))

nlltheta = [fn.NLL(u[:,i], sim, data) for i in range(len(thetas))]    

plt.plot(thetas, nlltheta,'-', label = r'fixed $\Delta m^2$ =  %s $eV^2$' % dm[0])
# plt.title(r'NLL($\theta$)')
plt.ylabel(r'NLL($\theta$)')
plt.xlabel(r'$\theta \times \pi$ (rad)')



dm =  2.0e-3
thetas = np.linspace(np.pi/8, 3*np.pi/8, 1000) #0 to pi/4 covers the whole range as it ~sin**(2theta) so bracket theta goes from 0 to pi/2
dm = np.array([dm] * len(thetas))
u = np.stack((thetas,dm))

nlltheta = [fn.NLL(u[:,i], sim, data) for i in range(len(thetas))]    

plt.plot(thetas, nlltheta,'-', label = r'fixed $\Delta m^2$ =  %s $eV^2$ ' % dm[0])
# plt.title(r'NLL($\theta$)')
plt.ylabel(r'NLL($\theta$)')
plt.xlabel(r'$\theta \times \pi$ (rad)')
plt.legend()

dm = 2.4e-3
thetas = np.linspace(np.pi/8, 3*np.pi/8, 1000) #0 to pi/4 covers the whole range as it ~sin**(2theta) so bracket theta goes from 0 to pi/2
dm = np.array([dm] * len(thetas))
u = np.stack((thetas,dm))

nlltheta = [fn.NLL(u[:,i], sim, data) for i in range(len(thetas))]    

plt.plot(thetas, nlltheta,'-', label = r'fixed $\Delta m^2$ =  %s $eV^2$ ' % dm[0])
# plt.title(r'NLL($\theta$)')
plt.ylabel(r'NLL($\theta$)')
plt.xlabel(r'$\theta$ (rad) ')
plt.axvline(x = np.pi/4, linestyle = '-.', color = 'black', label = r'$\theta = \frac{\pi}{4}$')

plt.legend()



