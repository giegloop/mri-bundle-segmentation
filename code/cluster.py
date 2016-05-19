import numpy as np
import networkx as nx
from scipy import spatial as sc
import itertools
import heapq
import time
from datetime import datetime
import pickle

########################################################################################################################
t_superp = 1 # temperature in superparamagnetic phase

t_iter = 1 # num. of iterations MC algorithm
t_burn_in = 0  # number of burn-in samples

q = 20  # num. of pot spin variables

k_neighbors = 20  # number of nearest neighbors

wm_threshold = 0.34  # threshold for white mass (FA > wm_threshold is considered white mass)

Gij_threshold = 0.5 # threshold for "core" clusters, section 4.3.2 of the paper

########################################################################################################################

start = time.time()


# load data with maximum diffusion and fractional anisotropy
max_diff = np.load("data/max_diff_sub.npy")
FA = np.load("data/FA_sub.npy")
size = np.load("data/dim_sub.npy")

x_dim = size[0]  # x dimension of original data
y_dim = size[1]  # y dimension of original data
z_dim = size[2]  # z dimension of original data

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


print("Start Monte Carlo for t_superp = {}...".format(t_superp)

Cij = np.zeros((N_points,N_points)) # probability of finding sites i and j in the same cluster

S = np.ones(N_points)  # Initialize S to ones

t_index = 0 # keep track of the burned-in samples
for i in range(t_iter):  # given iterations per temperature
    print("\t Iteration: {}/{}".format(i + 1, t_iter))
    G = nx.Graph()  # Initialize graph where we will store "frozen" bonds
    for i in range(N_points):
        G.add_node(i)

    for i in range(N_points):  # assign "frozen" bonds for neighbors
        neighbors = nearest_neighbors[i]  # nearest_neighbors has te be calculated in advance
        for j in neighbors:
            Jij = j_cost(i,j)
            pfij = (1 - np.exp(-Jij / t_superp)) if S[i] == S[j] else 0  # page 9 of the paper
            if np.random.uniform(0, 1) < pfij:
                G.add_edge(i, j)

    subgraphs = list(nx.connected_component_subgraphs(G))  # find SW-clusters
    for graph in subgraphs:
        new_q = np.random.randint(1, q+1)
        for node in graph.nodes():
            S[node] = new_q

    if t_index >= t_burn_in:
        for vi in range(N_points):
            for vj in range(vj, N_points):
                if (vi, vj) in G.edges(): # add 1 to count if nodes are connected (same SW-cluster)
                    Cij[vi][vj] = Cij[vi][vj] + 1

    t_index += 1

# average and obtain estimated probabilities
for vi in range(N_points):
    for vj in range(vj, N_points):
        Cij[vi][vj] = Cij[vi][vj] / t_iter

# calculate spin-spin correlation function Gij, (11) in the paper
Gij = np.zeros((N_points,N_points))
for vi in range(N_points):
    for vj in range(vj, N_points):
        Gij[vi][vj] = ((q-1)*Cij[vi][vj]+1)/q

# initialize graph where we are going to construct our final clustering
G = nx.Graph()

for i in range(N_points):
    G.add_node(i)

for i in range(N_points):
    for j in range(vj, N_points):
    	if Gij[i][j] > Gij_threshold
            G.add_edge(i, j)

# return final clustering
clusters = np.empty(N_points)
cluster_id = 1
for graph in list(nx.connected_component_subgraphs(G)):
    for node in graph.nodes()
        clusters[node] = cluster_id
    cluster_id += 1

# write results to file
results = {
    'x_dim': x_dim,
    'y_dim': y_dim,
    'z_dim': z_dim,
    'q': q,
    't_superp': t_superp,
    't_iter': t_iter,
    't_burn_in': t_burn_in,
    'k_neighbors': k_neighbors,
    'wm_threshold': wm_threshold,
    'Gij_threshold': Gij_threshold,
    'N_points': N_points,
    'clusters': clusters
}

f = open('results/clustering_' + '{:%d%m%y%H%M}'.format(datetime.now()) + '.pkl', 'wb')
pickle.dump(results, f)
f.close()

end = time.time()

print("Finished in {} seconds!".format(end - start))