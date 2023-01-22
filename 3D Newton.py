# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:33:17 2021

@author: Andre
"""

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

theta = np.pi/4
dm = 2.4e-3
L = 295

a = 1

u = [theta, dm, a]

newsim = fn.oscillated_sim3(sim,*u)

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
plt.title('Simulated data with trial parameters')

#%%
trying,count = fn.newton3(fn.nll3, np.pi/4, 2.4e-3,0.9,1e-4,threshold = 1e-4)
print(trying,count)
print(fn.nll3(*trying))
newtonerror = fn.nllerror3(*trying,thetafactor=0.08)
print(newtonerror)

#%%  3.1 Plotting data and simulated counts
data = np.loadtxt('observed.txt') #obserced mu
sim = np.loadtxt('simulated.txt') # simulated mu - unoscillated event rate prediction

theta = np.pi/4
dm = 2.4e-3
L = 295

a = 1

u = [theta, dm, a]
u = [0.7674447,  0.00253638, 1.035222]
u = trying
print(fn.nll3(*u,sim,data))

newsim = fn.oscillated_sim3(sim,*u)

bins = np.arange(0,10,0.05)

plt.subplot(1,2,1)
plt.hist(bins, bins, weights = data, label = r' Observed $\nu_{\mu}$ count') # treating each bin as a single points with a weight equal to its count
plt.legend()
plt.xlabel('Energy (GeV)')
plt.ylabel('Detection count')

plt.subplot(1,2,2)
plt.hist(bins, bins, weights = newsim, label = r'Simulated Annealing estimate counts') # treating each bin as a single points with a weight equal to its count
plt.legend()
plt.xlabel('Energy (GeV)')
plt.ylabel('Detection count')
# plt.title('Simulated Annealing estimate')

