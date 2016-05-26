"""
Create a plot of the susceptibility for phase analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

########################################################################################################################

id = str(2305162131)  # white mass > 0.5, 0 to 5
id = str(2505161633)  # white mass > 0, 2 to 5
id = str(2605161358)  # white mass > 0, 2.5 to 3.5
id = str(2605161412)

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

global_arg_max = np.argmax(pow(results['suscept_arr'], results['t_arr']) / results['N_points'])
print("Global maximum at {}".format(results['t_arr'][global_arg_max]))