"""
Create a plot of the susceptibility for phase analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
import colorsys
from mpl_toolkits.mplot3d import Axes3D

########################################################################################################################

id = str(1905162135)  # id of the results to use for analysis

########################################################################################################################

f = open('results/clustering_' + id + '.pkl', 'rb')
results = pickle.load(f)

size = np.load("data/dim_sub.npy")
x_dim = size[0]  # x dimension of original data
y_dim = size[1]  # y dimension of original data
z_dim = size[2]  # z dimension of original data

wm_threshold = results["wm_threshold"]  # threshold for white mass (FA > wm_threshold is considered white mass)
clusters = results["clusters"]

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
N_clusters = 20

HSV_tuples = [(x*1.0/N_clusters, 0.5, 0.5) for x in range(N_clusters)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
colors = np.array([RGB_tuples[int(i-1)] for i in clusters])

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
ax.scatter(x[clusters], y[clusters], z[clusters], c=colors[clusters])