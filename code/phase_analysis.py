"""
Create a plot of the susceptibility for phase analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

id = str(1905161121)

f = open('results/results_' + id + '.pkl', 'rb')
results = pickle.load(f)

# show susceptibility plot
plt.plot(results['t_arr'], pow(results['suscept_arr'], results['t_arr']) / results['N_points'])