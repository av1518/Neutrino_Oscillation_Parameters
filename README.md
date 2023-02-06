# A log-likelihood fit for extracting neutrino oscillation parameters

Experimental evidence of neutrino oscillations have commenced a vast new program of research seeking to answer questions regarding neutrino flavour masses and their role in the universe. It prompted the first major change tothe standard model in the last twenty years allowing for the purely quantum mechanical description in which neutrinos oscillate between flavours and have a non-zero mass. Due to the extremely long weak force interaction length, even the most sensitive neutrino detectors observe statistically limited numbers of neutrino interactions. To mitigate this limitation, statistical techniques have been largely employed which require computational minimisation algorithms to obtain results to a high degree of accuracy. The robustness and unique approaches of each method can significantly affect the results and their accuracies.

By constructing a negative log-likehood fit between the observed and expected detection counts, estimators for the neutrino oscillation parameter were extracted from the minimum point of the fit. Three distinct numerical minimisation methods were tested and deployed: Univariate method, Newton method and Simulated Annealing.




project_functions: Contains all the functions used in the poject. This file imported into all other python files
2D annealing: Simulated annealing for 2 parameter
2D Newton: Newton method + testing for 2 paramaters + contour plots for all other results
3D annealing: Simulated annealing for 3 parameters
3D Newton: Newton method for 3 paramers
3D univariate: univariate method in 3dimensions
plot for part 3.5: The plot for task 3.5 also plotted in report. It also contains the report figure 4
project_v2: follows the script tasks up to 3.5, also contains a lot of testing.


Note: the simulating annealing parts take about 20 minutes to run (on my computer). For the contour plot, I copied the results into a list in the 2D Newton method.
