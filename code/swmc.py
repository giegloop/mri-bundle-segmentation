"""
Svendsen-Wang Algorithm
"""

from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import networkx as nx
from scipy import spatial as sc
import itertools
import heapq
import time
from datetime import datetime
import pickle

########################################################################################################################

q = 20  # num. of pot spin variables

t_iter_per_temp = 200   # num. of iterations per temperature
t_burn_in = 50  # number of burn-in samples

k_neighbors = 10  # number of nearest neighbors

wm_threshold = 0.3  # threshold for white mass (FA > wm_threshold is considered white mass)

########################################################################################################################

start = time.time()
num_cores = multiprocessing.cpu_count()
print("Number of cores: {}", num_cores)

# load data with maximum diffusion and fractional anisotropy
max_diff = np.load("data/max_diff_sub.npy")
FA = np.load("data/FA_sub.npy")
size = np.load("data/dim_sub.npy")

# set size to dimensions of subset
(x_dim, y_dim, z_dim) = size

# create array of all possible (x,y,z) within subset
xyz = list(itertools.product(*[list(range(0, x_dim)), list(range(0, y_dim)), list(range(0, z_dim))]))

# remember indices of white matter in original dataset
wm_range = [i for i in range(x_dim * y_dim * z_dim) if FA[i] > wm_threshold]

# remove all non-white mass from dataset
max_diff = max_diff[FA > wm_threshold]
FA = FA[FA > wm_threshold]

# set dimension to length of reduced dataset
N_points = len(max_diff)


def index_to_xyz(i):
    return xyz[wm_range[i]]


# distance between two points in the lattice
def dist_lat(i, j):
    p1 = np.array(index_to_xyz(i))
    p2 = np.array(index_to_xyz(j))

    return sc.distance.euclidean(p1, p2)


# returns k-nearest neighbors with lattice distance within white matter
def wm_neighbors(i, k):
    dist = np.array([dist_lat(i, j) if i != j else np.float('infinity') for j in range(N_points)])
    return heapq.nsmallest(k, range(len(dist)), dist.take)


# computing nearest neighbors
def compute_nearest_neighbors(i):
    nn = []
    print("Computing nearest neighbors for {} of {}".format(i,N_points))
    for j in wm_neighbors(i, k_neighbors):
        if i < j:
            nn.append((i, j))
        else:
            nn.append((j, i))
    return nn

# computing nearest neighbors with parallel computation
print("Computing nearest neighbors...")
n = Parallel(n_jobs=num_cores)(delayed(compute_nearest_neighbors)(i) for i in range(N_points))

nn = set()
print("Merging the sets..")
for i in range(N_points):
    for j in n[i]:
        nn.add(j)

nn = list(nn)
N_neighbors = len(nn)

# compute average distance
print("Computing (average) distances between neighbors...")
nn_dist = np.array([dist_lat(i, j) for (i, j) in nn])
d_avg = np.mean(nn_dist)
dSq_avg = np.mean(pow(nn_dist, 2))


# Jij is the cost of considering coupled object i and object j. Here we use the
# maximum diffusion directions and the fractional anisotropy of each 3d pixel
# as a compression of the original data, as motivated from
# Diffusion Tensor MR Imaging and Fiber Tractography: Theoretic Underpinnings
# by P. Mukherjee, J.I. Berman, S.W. Chung, C.P. Hess R.G. Henry.
def j_cost(nn_index):
    (i, j) = nn[nn_index]
    vi = max_diff[i]
    vj = max_diff[j]
    j_shape = np.abs(np.dot(vi, vj) / (sc.distance.norm(vi) * sc.distance.norm(vj)))
    return j_shape

print("Computing Jij for all neighbors...")
nn_jij = np.array([j_cost(nn_index) for nn_index in range(N_neighbors)])

t_trans = (1 / (4 * np.log(1 + np.sqrt(q)))) * np.exp(-dSq_avg / 2 * pow(d_avg, 2))  # page 14 of the paper
t_ini = 0.05
t_end = 3
t_num_it = 100

print("Start Monte Carlo with t_start = {}, t_end = {}, steps = {}...".format(t_ini, t_end, t_num_it))

mag_arr = []  # array with average magnetation of each time
mag_sq_arr = []  # array with average squared magnetation of each time
t_arr = np.arange(t_ini, t_end, (t_end - t_ini) / t_num_it)  # range of times to loop over


def temp_step(t_i, t):
    print("Time: {}".format(t))
    S = np.ones(N_points)  # Initialize S to ones
    mag = 0
    magSq = 0
    t_index = 0

    for i in range(t_iter_per_temp):  # given iterations per temperature
        print("\t Iteration: {}/{}, time {}, {}/{}".format(i + 1, t_iter_per_temp, t, t_i, len(t_arr)))
        G = nx.Graph()  # Initialize graph where we will store "frozen" bonds
        for i in range(N_points):
            G.add_node(i)

        for nn_index, (i, j) in enumerate(nn):  # nearest_neighbors has te be calculated in advance
            pfij = (1 - np.exp(-nn_jij[nn_index] / t)) if S[i] == S[j] else 0  # page 9 of the paper
            if np.random.uniform(0, 1) < pfij:
                G.add_edge(i, j)

        subgraphs = list(nx.connected_component_subgraphs(G))  # find SW-clusters
        print("\t {} subgraphs".format(len(subgraphs)))
        for graph in subgraphs:
            new_q = np.random.randint(1, q+1)
            for node in graph.nodes():
                S[node] = new_q

        N_max = 0  # compute N_max, page 5 of the paper
        for q_val in range(q):
            new_N_max = sum(S == q_val)
            if new_N_max > N_max:
                N_max = new_N_max

        new_mag = (q * N_max - N_points) / ((q - 1) * N_points)  # (4) in paper
        if t_index >= t_burn_in:
            mag += new_mag
            magSq += pow(new_mag, 2)

        t_index += 1

    return mag / (t_iter_per_temp - t_burn_in), magSq / (t_iter_per_temp - t_burn_in)

mag_new_arr = Parallel(n_jobs=num_cores)(delayed(temp_step)(t_i, t) for t_i, t in enumerate(t_arr))
mag_arr = np.array([i[0] for i in mag_new_arr])
mag_sq_arr = np.array([i[1] for i in mag_new_arr])

# create susceptibility array
suscept_arr = (N_points / t_arr) * (mag_sq_arr - pow(mag_arr, 2))

# write results to file
results = {
    'x_dim': x_dim,
    'y_dim': y_dim,
    'z_dim': z_dim,
    'q': q,
    't_iter_per_temp': t_iter_per_temp,
    't_burn_in': t_burn_in,
    't_per_min': t_ini,
    't_per_max': t_end,
    't_num_it': t_num_it,
    'k_neighbors': k_neighbors,
    'wm_threshold': wm_threshold,
    'N_points': N_points,
    't_arr': t_arr,
    'suscept_arr': suscept_arr
}

id = '{:%d%m%y%H%M}'.format(datetime.now())
f = open('results/results_' + id + '.pkl', 'wb')
pickle.dump(results, f)
f.close()

end = time.time()

print("Finished in {} seconds!".format(end - start))
print("Exported with id {}".format(id))

# "COMPLETED" until the beginning of the page 14 of the paper
