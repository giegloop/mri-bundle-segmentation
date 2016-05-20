"""
Create a plot of the susceptibility for phase analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

########################################################################################################################

id = str(2005160620)  # id of the results to use for analysis

########################################################################################################################

f = open('results/results_' + id + '.pkl', 'rb')
results = pickle.load(f)

plt.figure(1)
ax1 = plt.subplot(211)
ax1.plot(results['t_arr'], results['suscept_arr'])
plt.title("Susceptibility")

ax2 = plt.subplot(212)
ax2.plot(results['t_arr'], pow(results['suscept_arr'], results['t_arr']) / results['N_points'])
plt.title("Susceptibility Density")