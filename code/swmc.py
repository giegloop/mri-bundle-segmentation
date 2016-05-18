import numpy as np
import networkx as nx
from scipy import spatial as sc
import itertools
import heapq
import matplotlib.pyplot as plt
import math as math

########################################################################################################################

x_dim = 41  # x dimension of original data
y_dim = 41  # y dimension of original data
z_dim = 91  # z dimension of original data

q = 20  # num. of pot spin variables

iter_perT = 50  # num. of iterations per temperature
t_per_min = 0.9  # min percentage from transition temperature
t_per_max = 1.1  # max percentage from transition temperature
t_etha = 0.999  # number of steps from min to max

k_neighbors = 20  # number of nearest neighbors

wm_threshold = 0.34  # threshold for white mass (FA > wm_threshold is considered white mass)

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


# distance between two points in the lattice
def dist_lat(i, j):
    p1 = np.array(index_to_xyz(i))
    p2 = np.array(index_to_xyz(j))

    return sc.distance.euclidean(p1, p2)


# returns k-nearest neighbours with lattice distance within white matter
def wm_neighbors(i, k):
    dist = np.array([dist_lat(i, j) for j in range(N_points)])
    return heapq.nsmallest(k, range(len(dist)), dist.take)


print("Computing nearest neighbours...")
nearest_neighbors = np.array([wm_neighbors(i, k_neighbors) for i in range(N_points)])


# returns average distance between all pairs i, j
def compute_d_avg():
    dists = np.empty(N_points)
    dists_sq = np.empty(N_points)
    for i in range(N_points):
        dist = np.array([dist_lat(i, j) for j in nearest_neighbors[i]])
        dists[i] = np.mean(dist)
        dists_sq[i] = np.mean(np.sqrt(dist))

    return np.mean(dists), np.mean(dists_sq)


# compute average distance
print("Counting average distance...")
d_avg, dSq_avg = compute_d_avg()

# cosine distance between two vectors from max_diff
def j_shape(i, j):
    v_i = max_diff[i]
    v_j = max_diff[j]

    return 2 - np.abs(np.dot(v_i, v_j) / (sc.distance.norm(v_i) * sc.distance.norm(v_j)))


# proximity function for two vectors from max_diff
def j_proximity(i, j):
    return 1 / k_neighbors * (dist_lat(i, j) / (2 * d_avg))


# Jij is the cost of considering coupled object i and object j. Here we use the
# maximum diffusion directions and the fractional anisotropy of each 3d pixel
# as a compression of the original data, as motivated from
# Diffusion Tensor MR Imaging and Fiber Tractography: Theoretic Underpinnings
# by P. Mukherjee, J.I. Berman, S.W. Chung, C.P. Hess R.G. Henry.
def j_cost(i, j):
    return j_proximity(i, j) * j_shape(i, j)

t_trans = (1 / (4 * np.log(1 + np.sqrt(q)))) * np.exp(-dSq_avg / 2 * pow(d_avg, 2))  # page 14 of the paper
t_ini = 1.1 * t_trans
t_end = 0.9 * t_trans

print("Start Monte Carlo with t_start = {}, t_end = {}, etha = {}...".format(t_ini, t_end, t_etha))

mag_arr = []  # array with average magnetation of each time
mag_sq_arr = []  # array with average squared magnetation of each time
t_arr = []  # array with average squared magnetation of each time


t_N = 1
t = t_ini * pow(t_etha, t_N)
while t > t_end:  # for each temperature
    print("Time: {}".format(t))
    S = np.ones(N_points)  # Initialize S to ones
    mag = 0
    magSq = 0

    for i in range(iter_perT):  # given iterations per temperature

        G = nx.Graph()  # Initialize graph where we will store "frozen" bonds
        for i in range(N_points):
            G.add_node(i)

        for i in range(N_points):  # assign "frozen" bonds for neighbors
            neighbors = nearest_neighbors[i]  # nearest_neighbors has te be calculated in advance
            for j in neighbors:
                Jij = j_cost(i,j)
                pfij = (1 - np.exp(-Jij / t)) if S[i] == S[j] else 0  # page 9 of the paper
                if np.random.uniform(0, 1) < pfij:
                    G.add_edge(i, j)

        subgraphs = list(nx.connected_component_subgraphs(G))  # find SW-clusters
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
        mag += new_mag
        magSq += pow(new_mag, 2)

    t_arr.append(t)
    mag_arr.append(mag / iter_perT)
    mag_sq_arr.append(magSq / iter_perT)

    t_N += 1
    t = t_ini * pow(t_etha, t_N)

suscept_arr = [(N_points / t_arr[t]) * (mag_sq_arr[t] - pow(mag_arr[t], 2)) for t in range(len(t_arr))]

t_arr = np.array(t_arr)
mag_arr = np.array(mag_arr)
mag_sq_arr = np.array(mag_sq_arr)
suscept_arr = np.array(suscept_arr)

plt.plot(suscept_arr)


print("Finished!")

# "COMPLETED" until the beginning of the page 14 of the paper
