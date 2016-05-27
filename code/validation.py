"""
Create a plot of the susceptibility for phase analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
import colorsys
import heapq
from mpl_toolkits.mplot3d import Axes3D

########################################################################################################################

id = str()

N_clusters_red = 6 # number of clusters to validate on

whitematter_covered = []
whitematter_val = []
score = []
wm_threshold_range = np.arange(0.25, 1, 0.01)
for wm_threshold_validation in wm_threshold_range:

    ########################################################################################################################

    f = open('results/clustering_' + id + '.pkl', 'rb')
    results = pickle.load(f)

    size = np.load("data/dim_sub.npy")
    x_dim = size[0]  # x dimension of original data
    y_dim = size[1]  # y dimension of original data
    z_dim = size[2]  # z dimension of original data

    wm_threshold = results["wm_threshold"]  # threshold for white mass (FA > wm_threshold is considered white mass)
    clusters = results["clusters"]

    # load data with maximum diffusion and fractional anisotropy
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
    N_clusters = len(np.unique(clusters))

    # load data with maximum diffusion and fractional anisotropy for validation
    max_diff_val = np.load("data/max_diff_sub.npy")
    FA_val = np.load("data/FA_sub.npy")

    # save some values for recovering xyz values by index of reduced dataset
    wm_range_val = [i for i in range(x_dim * y_dim * z_dim) if FA_val[i] > wm_threshold_validation]

    # remove non-white mass from dataset for validation
    max_diff = max_diff[FA > wm_threshold_validation]
    FA = FA[FA > wm_threshold_validation]

    count = np.array([len(clusters[clusters==i]) for i in range(N_clusters)])
    count_indexes = heapq.nlargest(N_clusters_red, range(len(count)), count.take)
    plot_indexes = np.array([i in count_indexes for i in clusters])

    wm_range = np.array(wm_range)
    points_in_clusters = wm_range[plot_indexes]

    whitematter_val.append(len(wm_range_val))
    whitematter_covered.append(len(set(points_in_clusters).intersection(set(wm_range_val))))

score = np.array(whitematter_val) / np.array(whitematter_covered)

line1 = plt.plot(list(wm_threshold_range), whitematter_val)
line2 = plt.plot(list(wm_threshold_range), whitematter_covered)
plt.legend(['total # of white matter', '# of white matter covered by 6 largest clusters'])
plt.xlim([0.25, 1])
plt.xlabel("FA threshold for total # of white matter")