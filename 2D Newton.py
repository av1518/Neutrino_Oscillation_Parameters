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

def sinc(x,y):
    return np.sin(np.sqrt(x*x + y*y))/np.sqrt(x*x+y*y)

print(fn.newton(sinc, 1,1,1e-4)) #test function =>: expect 0,0 and returned something  very close to it

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
trying,count = fn.newton(fn.nll, np.pi/4, 2.4e-3, 1e-4,threshold = 1e-9)
print('Newton method estimates =', trying,count)
print(fn.nll(*trying))
print(fn.nll(*trying))
#%% contour plot
annealsol = [0.7854766920326939, 0.002390671871475111]
thetas = np.linspace(annealsol[0]-0.17, annealsol[0]+0.17,300)
dms = np.linspace(annealsol[1]-0.00035, annealsol[1]+0.00035,300)
X, Y = np.meshgrid(thetas,dms)
F = fn.nll(X, Y)

unisol = [0.7676360620971523, 0.002389524666604339]
unierror = fn.nllerror(*unisol)
newtonsol = [trying[0], trying[1]]
newtonerror = fn.nllerror(*newtonsol, thetafactor = 0.04)
annealsol = [0.7849673032052427, 0.0023928111460407666]
annealerror = fn.nllerror(*annealsol, thetafactor = 0.04)
print(fn.nll(*unisol))
print(fn.nll(*newtonsol))
print(fn.nll(*annealsol))
print(f'Newton method errors = {newtonerror}')

annealstd = [0.020383999350713958, 0.020383999350713958 , 3.0240584043478854e-05, 3.0240584043478854e-05]
annealerrortotal = []
for i,j in zip(annealerror,annealstd):
    b = np.sqrt(i*i + j*j)
    annealerrortotal.append(b)
annealerror = annealerrortotal

#%%


plt.contour(X,Y,F, 15)
plt.xlabel(r'$\theta_{23}$')
plt.ylabel(r'$\Delta m^2_{23}$')
plt.plot(*newtonsol, 'x', label = 'Newton', color = 'green', markersize = 7 )
plt.plot(*annealsol, 'x', label = 'Annealing', color = 'black')
plt.plot(*unisol, 'x', label = 'Univariate', color = 'red')
xerr=np.array([[unierror[0] ,unierror[1]]])
plt.errorbar(unisol[0], unisol[1], xerr=[[unierror[0]] ,[unierror[1]]], yerr = [[unierror[2]] ,[unierror[3]]], color = 'red', label = 'Univariate error', capsize = 2)
plt.errorbar(newtonsol[0],newtonsol[1], xerr = [[newtonerror[0]],[newtonerror[1]]], yerr = [[newtonerror[2]],[newtonerror[3]]], color = 'green', label = 'Newton error', capsize = 2, elinewidth = 2.1)
plt.errorbar(annealsol[0],annealsol[1], xerr = [[annealerror[0]],[annealerror[1]]], yerr = [[annealerror[2]],[annealerror[3]]], color = 'black', label = 'Annealing error', capsize = 2)
plt.legend()
plt.xlim(annealsol[0]-0.044, annealsol[0]+0.044)
plt.ylim(annealsol[1]-0.00007,annealsol[1]+0.00007)
plt.colorbar()
plt.show()
#%%
# plt.xlim(annealsol[0]-0.15, annealsol[0]+0.15)
# plt.ylim(0.0022,0.0026)