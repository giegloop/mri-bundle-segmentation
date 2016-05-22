"""
Creates a scatterplot for the current subset
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

########################################################################################################################

size = np.load("data/dim_sub.npy")
x_dim = size[0]  # x dimension of original data
y_dim = size[1]  # y dimension of original data
z_dim = size[2]  # z dimension of original data

wm_threshold = 0.5  # threshold for white mass (FA > wm_threshold is considered white mass)

########################################################################################################################


# load data with maximum diffusion and fractional anisotropy
# max_diff = np.genfromtxt('data/embeddings', dtype='float64')
# FA = np.genfromtxt('data/FA', dtype='float64')
max_diff = np.load("data/max_diff_sub.npy")
FA = np.load("data/FA_sub.npy")

# save some values for recovering xyz values by index of reduced dataset
xyz = list(itertools.product(*[list(range(0, x_dim)), list(range(0, y_dim)), list(range(0, z_dim))]))
wm_range = [i for i in range(x_dim * y_dim * z_dim) if FA[i] > wm_threshold]

# remove all non-white mass from dataset
max_diff = max_diff[FA > wm_threshold]
FA = FA[FA > wm_threshold]

# set dimension to length of reduced dataset
N_points = len(max_diff)

def index_to_xyz(i):
    return xyz[wm_range[i]]

x = np.empty(N_points)
y = np.empty(N_points)
z = np.empty(N_points)
for i in range(N_points):
    index = index_to_xyz(i)
    x[i] = index[0]
    y[i] = index[1]
    z[i] = index[2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

# plt.scatter(y, z)