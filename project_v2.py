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
#%%  3.1 Plotting data and simulated counts
data = np.loadtxt('observed.txt') #obserced mu
sim = np.loadtxt('simulated.txt') # simulated mu - unoscillated event rate prediction

bins = np.arange(0,10,0.05)

plt.subplot(2,1,1)
plt.hist(bins, bins, weights = data, label = r'Observed $\nu_{\mu}$ data') # treating each bin as a single points with a weight equal to its count
plt.legend()
plt.xlabel('Energy (GeV)')
plt.ylabel('Counts')

plt.subplot(2,1,2)
plt.hist(bins, bins, weights = sim, label = r' Unoscillated $\nu_{\mu}$ count') # treating each bin as a single points with a weight equal to its count
plt.legend()
plt.xlabel('Energy (GeV)')
plt.ylabel('Counts')

#%% 3.2 Fit function: Coding probability P, varying paramters to see its behaviour

'Testing initial trial:'
E = np.linspace(0.05, 10.05, 500)
theta = np.pi/4
dm = 2.4e-3
L = 295
# theta = np.arange(0,2 * np.pi, np.pi/4)
prob = [fn.P(i, theta, dm, L) for i in E]
plt.plot(E,prob, '.-')
plt.title('Trial Probability')
plt.ylabel(r'Probability $P(\nu_{\mu} \rightarrow \nu_{\mu})$')
plt.xlabel('Energy (GeV)')
#%% Varying theta
theta = np.arange(2*np.pi/16, 5*np.pi/16, np.pi/16  )
E = np.linspace(0.05, 10.05, 500)
dm = 2.4e-3
# def P(E, theta, dm_sq, L):
#     return  np.sin(2*theta)**2 * np.sin((1.267 * dm_sq * L)/E)**2

probs_theta = []
for j in theta:
    vals = []
    for i in E:
        probability = fn.P(i,j, dm, L)
        vals.append(probability)
    probs_theta.append(vals)
probs_theta = np.array(probs_theta)

labels = theta/np.pi
labels = [r'$\frac{\pi}{8}$ rad',r'$\frac{3\pi}{16}$ rad',r'$\frac{\pi}{4}$ rad']

# plt.subplot(2,1,1)
for i in range(len(probs_theta)):
    # plt.subplot(1,len(probs_theta), i+1)
    plt.plot(E,probs_theta[i,:], '-', label = r'$\theta$ =  %s ' % labels[i] )
plt.legend()
plt.title(r'Varying $\theta$ with constant $\Delta m^2 = 2.4 \times 10^{-3} eV^2$')

'change this to get the x limits'

plt.ylabel(r'Probability $P(\nu_{\mu} \rightarrow \nu_{\mu})$')
plt.xlabel('Energy (GeV)')
# plt.xlim(0,2) # Change this to get the xlimits

#%%
theta = np.pi/4

dm = np.arange(1e-4, 10e-4, 1e-4)
dm  = (1e-2,10e-4,1e-4)
E = np.linspace(0.05, 10.05, 500)

probs_dm = []
for j in dm:
    vals = []
    for i in E:
        probability = fn.P(i,theta, j, L)
        vals.append(probability)
    probs_dm.append(vals)
probs_dm = np.array(probs_dm)
# plt.subplot(2,1,2)
for i in range(len(probs_dm)):
    # plt.subplot(1,len(probs_theta), i+1)
    plt.plot(E,probs_dm[i,:], '-', label = r'$\Delta m^2$ =  %s $eV^{2}$' % dm[i]  )
plt.legend()
plt.title(r'Varying $\Delta m ^{2}_{23}$ with constant $\theta_{23} = \frac{\pi}{4}$ rad')

plt.ylabel(r'Probability $P(\nu_{\mu} \rightarrow \nu_{\mu})$')
plt.xlabel('Energy (GeV)')
'change this to get the x limits'
# plt.xlim(0,2) # Change this to get the xlimits
#%% 3.2 Appling Probability P onto simulated data with trial parameters
# E = np.linspace(0.05, 10.05, 500)

# theta_min = 0.7641005758007319
theta = np.pi/4
dm = 2.4e-3
L = 295
u = [theta, dm]
# u = [0.7675464606410061, 0.002388735960253954]
# u = [0.7853981633974483, 0.002366388255185324]
u = [0.7677570033447106, 0.002388736046007897]

newsim = fn.oscillated_sim(sim,*u)

bins = np.arange(0,10,0.05)

plt.subplot(1,2,1)
plt.hist(bins, bins, weights = data, label = r'Experimental $\nu_{\mu}$ count') # treating each bin as a single points with a weight equal to its count
plt.legend()
plt.xlabel('Energy (GeV)')
plt.ylabel('Counts')

plt.subplot(1,2,2)
plt.hist(bins, bins, weights = newsim, label = r'Simulated $\nu_{\mu}$ count') # treating each bin as a single points with a weight equal to its count
plt.legend()
plt.xlabel('Energy (GeV)')
plt.ylabel('Counts')
#%% 3.3 Coding NLL function, plotting for fixed trial dm, plotting NLL(theta)
'''
u = [theta, dm**2] -> dm**2 is dm in this script
'''

dm = 2.4e-3
thetas = np.linspace(0, np.pi/4, 1000) #0 to pi/4 covers the whole range as it ~sin**(2theta) so bracket theta goes from 0 to pi/2
dm = np.array([dm] * len(thetas))
u = np.stack((thetas,dm))

nlltheta = [fn.NLL(u[:,i], sim, data) for i in range(len(thetas))]    

plt.plot(thetas, nlltheta,'-', label = r'fixed $\Delta m^2$ =  %s ' % dm[0])
plt.title(r'NLL($\theta$)')
plt.ylabel(r'NLL($\theta$)')
plt.xlabel(r'$\theta \times \pi$ (rad)')
plt.legend()
# plt.xlim(0.76,0.785)

print('Approximate position of minimum(by eye) =', 0.78)
#%% 3.4 Parabolic minimiser TESTING
'Testin the parabola search on simple function '
def y(x):
    return x*x + 2*x + 3

x = np.linspace(-2000,1000,100)
plt.plot(x,y(x))

test,count,dx,x_last,y_last = fn.parabmin(y,-994.888888444,0.33222, 0.0001)
test,count,dx,x_last,y_last = fn.parabmin(y,-994.888888444,-50, 0.0001)
print(test,'count = ',count, 'dx =', dx)
#%%another one
def gauss(x):
    return -1*np.exp(-1*(x-5)**2) 

x = np.linspace(0,10,100)
plt.plot(x,gauss(x))
test,count,dx,x_last,y_last = fn.parabmin(gauss,4.5,4.6, 0.00001)
print(test,'count = ',count, 'dx =', dx)

#%%

def gauss(x):
    gauss1 = -1*np.exp(-1*(x-5)**2)
    gauss2 = -2*np.exp(-1*(x-3)**2)
    return gauss1 + gauss2

x = np.linspace(0,10,100)
plt.plot(x,gauss(x))
# test,count,dx,x_last,y_last = fn.parabmin(gauss,4.5,4.6, 0.00001)
# print(test,'count = ',count, 'dx =', dx)



#%% minimizing the NLL(theta)
'''
function we want to minimize: NLL(u[theta, dm**2]
'''
        
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
# plt.subplot(1,2,1)
plt.plot(thetas, nlltheta,'-', label = r'fixed $\Delta m^2$ =  %s ' % dm[0])
plt.axvline(x = theta_min, linestyle = '-.',label = r'$\theta_{min}$ from parabolic search',  )
plt.title(r'NLL($\theta$)')
plt.ylabel(r'NLL($\theta$)')
plt.xlabel(r'$\theta$')
plt.legend()
#%% 3.5 Accurazy of fit result -> secand method
'''
want to code root finding problem:
    NLL(thetaprime) - b = 
    where b = NLL_theta(theta_min) + 0.5
    
'''
dm = 2.4e-3
thetas = np.linspace(0, np.pi/4, 1000) #0 to pi/4 covers the whole range as it ~sin**(2theta) so bracket theta goes from 0 to pi/2
dm = np.array([dm] * len(thetas))
u = np.stack((thetas,dm))

def nllroot(theta):
    b = NLL_theta(theta_min) + 0.5 
    return NLL_theta(theta) - b

leftroot, count = fn.secand(nllroot, theta_min, theta_min +0.001)
print('root from secand = ', leftroot, count)

plt.plot(thetas, nlltheta,'-', label = r'fixed $\Delta m^2$ =  %s ' % dm[0])
plt.axvline(x = theta_min, linestyle = '-.',label = r'$\theta_{min}$ from parabolic search',  )
plt.title(r'NLL($\theta$)')
plt.ylabel(r'NLL($\theta$)')
plt.xlabel(r'$\theta \times \pi$ (rad)')
plt.xlim(leftroot-0.03, np.pi/4)
plt.ylim(NLL_theta(theta_min)-0.25/4,NLL_theta(leftroot))
plt.axvline(x=leftroot, label = 'Leftroot', color = 'red')
plt.legend()

rightroot, count = fn.secand(nllroot, np.pi/4, np.pi -0.001) #cannot get the right root using secand
std = abs(rightroot-theta_min)
print('root from secand, iterations= ', leftroot, count)
print(f'error from secand method = {std}')

# '''trying function update'''
# xmin,error = fn.parabmin(NLL_theta, 0.6, np.pi/4, 1e-7, extra = 0)
# print(f'theta_min = {xmin} +/- {error}')

#%%
dm = 2.4e-3
test,std = fn.minimize_theta(np.pi/4,dm,sim,data)
print(test,std)


#%% Trying the bisection method
leftroot, leftcount = fn.bisection(nllroot, 0.710, theta_min, tol = 1e-9)
print('leftroot from bisection =', leftroot, leftcount)
leftdiff = abs(leftroot - theta_min)
print('Difference of NLL(thetaroot) - NLL(theta_min) =', NLL_theta(leftroot) - NLL_theta(theta_min))

right, rightcount = fn.bisection(nllroot, theta_min,np.pi/4, tol = 1e-9)
print('rightroot from bisection =', rightroot, rightcount)
rightdiff = abs(rightroot - theta_min)
print('Difference of NLL(thetaroot) - NLL(theta_min) =', NLL_theta(rightroot) - NLL_theta(theta_min))

std = (leftdiff + rightdiff)/2
print(f'standard div = {std}')
print(f'Theta+ = {rightdiff}, theta- = {leftdiff}')

plt.subplot(2,2,1)
plt.plot(thetas, nlltheta,'-', label = r'fixed $\Delta m^2$ =  %s ' % dm)
plt.axvline(x = theta_min, linestyle = '-.',label = r'$\theta_{min}$ from parabolic search',  )
plt.title(r'NLL($\theta$)')
plt.ylabel(r'NLL($\theta$)')
plt.xlabel(r'$\theta \times \pi$ (rad)')
plt.xlim(leftroot-0.001, np.pi/4)
plt.ylim(NLL_theta(theta_min)-0.25/4,NLL_theta(leftroot)+0.25/4)
plt.axvline(x=leftroot, label = 'Leftroot', color = 'red')
plt.axvline(x = rightroot, label = 'Rightroot', color = 'red')
plt.legend()

#%% Getting error using second derivative of latest parapola in parabolic search for minimum
x  = x_latest
y = y_latest
# x = [0.6, (0.6+np.pi/4)/2, np.pi/4]
y = [fn.nll(i,dm,sim,data) for i in x]

A = y[0] / ((x[0] - x[1]) * (x[0] - x[2]))
B = y[1] / ((x[1] - x[0]) * (x[0] - x[2]))
C = y[2] / ((x[2] - x[0]) * (x[2] - x[1]))

Psecder = 2*(A+B+C)
print(Psecder)
std_an = np.sqrt(1/Psecder)
print(std_an)
#%% 

x = sorted(x_latest)
y = [fn.nll(i,dm,sim,data) for i in x]

A = y[0] / ((x[0] - x[1]) * (x[0] - x[2]))
B = y[1] / ((x[1] - x[0]) * (x[0] - x[2]))
C = y[2] / ((x[2] - x[0]) * (x[2] - x[1]))

Psecder = 2*(A+B+C)
print(Psecder)
std_an = np.sqrt(1/Psecder)
print(std_an)

#%% 4 Univariate method: 
'''
Minimizing dm now: using fixed theta of theta_min calculated above
u = [theta, dm**2] -> dm**2 is dm in this script
'''
#%% Plotting sin**2(1.267dmL/E) for fixed L,E to find range of dm
def func(dm):
    L = 295
    E = 5
    return np.sin(1.267 * dm * L/E)**2

L = 295
E = 5
dm = np.linspace(0,0.1,1000)
plt.plot(dm, func(dm))
maxdm = np.pi * E/(2 * 1.267 * L)
print(maxdm)
plt.axvline(x = maxdm) 
#%% Plotting NLL for range of dm values
dms = np.linspace(0,0.02,1000)
theta_trial = np.pi/4 #trial value
thetas = np.array([theta] * len(dms))
u = np.stack((thetas,dms))

nlldm = [fn.NLL(u[:,i], sim, data) for i in range(len(dms))]

plt.plot(dms, nlldm,'-', label = r'fixed $\theta $ =  %s ' % thetas[0])
plt.title(r'NLL($\Delta m ^2$)')
plt.ylabel(r'NLL($\Delta m ^2$)')
plt.xlabel(r'$\Delta m^2$')
plt.legend()
# plt.xlim(0.76,0.785)

# def NLL_theta(theta, dm = 2.4e-3): #defining NLL just for theta varying to use in minimiser
#     # dm = 2.4e-3
#     u = [theta,dm]
#     nllvalue = fn.NLL(u,sim,data)
#     return nllvalue

def NLL_dm(dm, theta = theta_trial): #defining NLL just for dm varying to use in minimiser
    u = [theta,dm]
    nllvalue = fn.NLL(u,sim,data)
    return nllvalue
#%%by eye
dm_min ,count,dx,x_latest,y_latest = fn.parabmin(NLL_dm, 0.00,0.0025, 1e-5)
print(r'Value of dm_min from parabolic optimisation = %a, \
      Number of iterations = %a  \
          last change in x = %a  \
              Last 3 x values in search = %a \
                  last 3 y values in search = %a'
          % (dm_min, count, dx, x_latest,y_latest)) 

plt.plot(dms, nlldm,'-', label = r'fixed $\theta $ =  %s ' % thetas[0])
plt.axvline(x = dm_min, linestyle = '-.',label = r'$dm_{min}$ from parabolic search',  )
plt.title(r'NLL($dm$)')
plt.ylabel(r'NLL($dm$)')
plt.xlabel(r'dm')
plt.xlim(0.001,0.004)
plt.legend()
#%% Using index(min(list)) => not sure if i'm allowed to
dm_guess = dms[nlldm.index(min(nlldm))]

dm_min ,count,dx,x_latest,y_latest = fn.parabmin(NLL_dm, dm_guess-0.0025/4, dm_guess+0.0025/4, 1e-5)
print(r'Value of dm_min from parabolic optimisation = %a, \
      Number of iterations = %a  \
          last change in x = %a  \
              Last 3 x values in search = %a \
                  last 3 y values in search = %a'
          % (dm_min, count, dx, x_latest,y_latest)) 

plt.plot(dms, nlldm,'-', label = r'fixed $\theta $ =  %s ' % thetas[0])
plt.axvline(x = dm_min, linestyle = '-.',label = r'$dm_{min}$ from parabolic search',  )
plt.title(r'NLL($dm$)')
plt.ylabel(r'NLL($dm$)')
plt.xlabel(r'dm')
plt.xlim(0.001,0.004)
plt.legend()

#%% Testing minimize_param functions
theta = np.pi/4
dm = 2.4e-3
u_in = [theta,dm]

theta_test = fn.minimize_theta(*u_in, sim, data)
print(theta_test)
print(theta_min)

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


#%% Find the error and potting for theta
theta_min = u_thetafirst[0]
dm_min = u_thetafirst[1]

thetas = np.linspace(0,1.5* np.pi/4, 1000)


def nllroot_theta(theta):
    b = NLL_theta(theta_min) + 0.5 
    return NLL_theta(theta) - b
        
def NLL_theta(theta, dm = dm_min): #defining NLL just for theta varying to use in minimiser
    # dm = 2.4e-3
    u = [theta,dm]
    nllvalue = fn.NLL(u,sim,data)
    return nllvalue

nlltheta = [NLL_theta(i) for i in thetas]

leftroot, leftcount = fn.bisection(nllroot_theta, 0.710, u_thetafirst[0], tol = 1e-9)
print('leftroot from bisection =', leftroot, leftcount)
leftdiff = abs(leftroot - theta_min)
print('Difference of NLL(thetaroot) - NLL(theta_min) =', NLL_theta(leftroot) - NLL_theta(theta_min))

rightroot, rightcount = fn.bisection(nllroot_theta, theta_min,theta_min+0.02, tol = 1e-9)
print('rightroot from bisection =', rightroot, rightcount)
rightdiff = abs(rightroot - theta_min)
print('Difference of NLL(thetaroot) - NLL(theta_min) =', NLL_theta(rightroot) - NLL_theta(theta_min))

std = (leftdiff + rightdiff)/2
print(f'standard div = {std}')

plt.plot(thetas, nlltheta,'-', label = r'fixed $\Delta m^2$ =  %s ' % dm_min)
plt.axvline(x = theta_min, linestyle = '-.',label = r'$\theta_{min}$ from parabolic search',  )
plt.title(r'NLL($\theta$)')
plt.ylabel(r'NLL($\theta$)')
plt.xlabel(r'$\theta \times \pi$ (rad)')
plt.xlim(leftroot-0.001, rightroot+0.001)
plt.ylim(NLL_theta(theta_min)-0.25/4,NLL_theta(leftroot)+0.25/4)
plt.axvline(x=leftroot, label = 'Leftroot', color = 'red')
plt.axvline(x = rightroot, label = 'Rightroot', color = 'red')
plt.legend()

#%% Find the error and potting for theta
theta_min = u_thetafirst[0]
dm_min = u_thetafirst[1]

dms = np.linspace(0,0.01,1000)


def NLL_dm(dm, theta_val = theta_min): #defining NLL just for dm varying to use in minimiser
    u = [theta,dm]
    nllvalue = fn.NLL(u,sim,data)
    return nllvalue
        
def nllroot_dm(dm):
    b = NLL_dm(dm_min) + 0.5
    return NLL_dm(dm) - b

nlldms = [NLL_dm(i) for i in dms]


leftroot, leftcount = fn.bisection(nllroot_dm, 0.0022, u_thetafirst[1], tol = 1e-9)
print('leftroot from bisection =', leftroot, leftcount)
leftdiff = abs(leftroot - dm_min)
print('Difference of NLL(dmroot) - NLL(dm_min) =', NLL_dm(leftroot) - NLL_dm(dm_min))

rightroot, rightcount = fn.bisection(nllroot_dm, dm_min,0.0025, tol = 1e-9)
print('rightroot from bisection =', rightroot, rightcount)
rightdiff = abs(rightroot - dm_min)
print('Difference of NLL(dmroot) - NLL(dm_min) =', NLL_dm(rightroot) - NLL_dm(dm_min))

std = (leftdiff + rightdiff)/2
print(f'standard div = {std}')
plt.plot(dms, nlldms,'-', label = r'fixed $\theta_{min}$ =  %s ' % theta_min)
plt.axvline(x = dm_min, linestyle = '-.',label = r'$\delta m_{min}$ from parabolic search',  )
plt.title(r'NLL($\Delta m$)')
plt.ylabel(r'NLL($\Delta m$)')
plt.xlabel(r'$\Delta m$')
plt.xlim(leftroot-0.0001, rightroot+0.0001)
# plt.ylim(NLL_dm(dm_min)-0.25/4,NLL_dm(leftroot)+0.25/4)
plt.axvline(x=leftroot, label = 'Leftroot', color = 'red')
plt.axvline(x = rightroot, label = 'Rightroot', color = 'red')
plt.legend()

