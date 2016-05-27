"""
Create a plot of the susceptibility for phase analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

########################################################################################################################

id = str()  # white mass > 0.5, 0 to 5

########################################################################################################################

f = open('results/results_' + id + '.pkl', 'rb')
results = pickle.load(f)

fig = plt.figure(1)

ax1 = plt.subplot(3,1,1)
ax1.plot(results['t_arr'], results['mag_arr'])
plt.title("Magnetization")
plt.xlabel("T")
plt.ylabel("<$m$>")
plt.xlim(0,1)
plt.grid(True)

ax1 = plt.subplot(3,1,2)
ax1.plot(results['t_arr'], results['suscept_arr'])
plt.title("Susceptibility")
plt.xlabel("T")
plt.ylabel("$\chi$")
plt.xlim(0,1)
plt.grid(True)

ax2 = plt.subplot(3,1,3)
ax2.plot(results['t_arr'], pow(results['suscept_arr'], results['t_arr']) / results['N_points'])
plt.title("Susceptibility Density")
plt.ylabel("$\chi^T / N$")
plt.xlabel("T")
plt.xlim(0,1)
plt.grid(True)

# fig.set_size_inches(18.5, 10.5, forward=True)
plt.tight_layout(pad=2, w_pad=0, h_pad=0)
fig.savefig('phase_analysis.eps', format='eps', dpi=1000)

global_arg_max = np.argmax(pow(results['suscept_arr'], results['t_arr']) / results['N_points'])
print("Global maximum at {}".format(results['t_arr'][global_arg_max]))