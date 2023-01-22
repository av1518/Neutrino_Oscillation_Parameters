import numpy as np
import matplotlib.pyplot as plt
import project_functions as fn

plot_params = {'axes.labelsize':12,
          'axes.titlesize':12,
          'font.size':12,
          'figure.figsize':[10,4]}
plt.rcParams.update(plot_params)
plt.rcParams.update(plot_params)
#%%  3.1 Plotting data and simulated counts
data = np.loadtxt('observed.txt') #obserced mu
sim = np.loadtxt('simulated.txt') # simulated mu - unoscillated event rate prediction

bins = np.arange(0,10,0.05)

plt.subplot(1,2,1)
plt.hist(bins, bins, weights = data, label = r'Experimental $\nu_{\mu}$ count') # treating each bin as a single points with a weight equal to its count
plt.legend()
plt.xlabel('Energy (GeV)')
plt.ylabel('Counts')

plt.subplot(1,2,2)
plt.hist(bins, bins, weights = sim, label = r'Simulated $\nu_{\mu}$ count') # treating each bin as a single points with a weight equal to its count
plt.legend()
plt.xlabel('Energy (GeV)')
plt.ylabel('Counts')
#%%testing finite difference method fmd1st:
    
def f(x,y):
    return x**2 + y**2 + 3*x + 2*y

test = fn.f_x(f,5,6,0.002)
testnd = fn.f_xx(f, 5, 6, 0.00002)
testx = fn.f_x(f, 5, 6, 0.0002,1)

def f(x,y):
    return x**3 + (x**3) * (y**2) + x 

print(fn.f_xy(f, 5, 6, 0.00002, 0.00002))

#%% Testing the hessian using the example on minimisation notes pg 124
def f(x,y):
    return x**2 + 2*y**2 + x*y + 3*x

hessian = fn.H(f,0,0,0.0002)
# print(hessian)
invhessian = np.linalg.inv(hessian)
# print(invhessian)

testing = fn.newton(f,2,2,1e-4,threshold = 1e-9)
print(testing) #works

#%% trying on NLL
trying,count = fn.newton(fn.nll, 0.7, 2.4e-3, 1e-4,threshold = 1e-9)
print(trying,count)
print(fn.nll(*trying))



# nll = nll)
# plt.plot(theta,nll)