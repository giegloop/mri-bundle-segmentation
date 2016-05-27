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
id = str(2605161436)  # normal cost function (1 - np.abs...
id = str(2605161646)  # normal cost function (1 - np.abs... zoomed
id = str(2605162042)  # reversed cost function (np.abs...
id = str(2605162145)  # reversed cost function (np.abs... zoomed
id = str(2705161259)  # new cost function (np.exp(-6 * pow(j_shape, 2))
id = str(2705161428)  # new cost function (np.exp(-6 * pow(j_shape, 2)) zoomed

########################################################################################################################

f = open('results/results_' + id + '.pkl', 'rb')
results = pickle.load(f)

plt.figure(1)

ax1 = plt.subplot(3,1,1)
ax1.plot(results['t_arr'], results['mag_arr'])
plt.title("Magnetization")
plt.grid(True)

ax1 = plt.subplot(3,1,2)
ax1.plot(results['t_arr'], results['suscept_arr'])
plt.title("Susceptibility")
plt.grid(True)

ax2 = plt.subplot(3,1,3)
ax2.plot(results['t_arr'], pow(results['suscept_arr'], results['t_arr']) / results['N_points'])
plt.title("Susceptibility Density")
plt.grid(True)

global_arg_max = np.argmax(pow(results['suscept_arr'], results['t_arr']) / results['N_points'])
print("Global maximum at {}".format(results['t_arr'][global_arg_max]))