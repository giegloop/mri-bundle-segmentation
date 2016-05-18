"""
Create subset of the dataset
"""

import numpy as np
import networkx as nx
from scipy import spatial as sc
import itertools

max_diff_full = np.genfromtxt('data/embeddings', dtype='float64')
FA_full = np.genfromtxt('data/FA', dtype='float64')

# dimensions of original data
x_dim = 210
y_dim = 210
z_dim = 130

# list to convert i to xyz
xyz = list(itertools.product(*[list(range(0, x_dim)), list(range(0, y_dim)), list(range(0, z_dim))])) # i to xyz

# dimension of wanted subset
x_sub = [40, 80]
y_sub = [40, 80]
z_sub = [40, 130]

x_dim_sub = x_sub[1] - x_sub[0] + 1
y_dim_sub = y_sub[1] - y_sub[0] + 1
z_dim_sub = z_sub[1] - z_sub[0] + 1
N_sub = x_dim_sub * y_dim_sub * z_dim_sub

max_diff_sub = np.empty((N_sub, 3))
FA_sub = np.empty(N_sub)
a = 0
for x in range(x_sub[0],x_sub[1]+1):
    print("x: {}".format(x))
    for y in range(y_sub[0], y_sub[1]+1):
        for z in range(z_sub[0], z_sub[1]+1):
            i = x_dim*(x)+y_dim*(y)+z
            max_diff_sub[a] = max_diff_full[i]
            FA_sub[a] = FA_full[i]
            a += 1

np.save("data/max_diff_sub", max_diff_sub)
np.save("data/FA_sub", FA_sub)