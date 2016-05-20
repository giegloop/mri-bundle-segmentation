import numpy as np
import networkx as nx
from scipy import spatial as sc
import itertools
import heapq
import time
from datetime import datetime
import pickle

########################################################################################################################

t_superp = 0.05 # temperature in superparamagnetic phase

t_iter = 500 # num. of iterations MC algorithm
t_burn_in = 50  # number of burn-in samples

q = 20  # num. of pot spin variables
k_neighbors = 20  # number of nearest neighbors
wm_threshold = 0.5  # threshold for white mass (FA > wm_threshold is considered white mass)
Gij_threshold = 0.5 # threshold for "core" clusters, section 4.3.2 of the paper

########################################################################################################################

start = time.time()

# load data with maximum diffusion and fractional anisotropy
max_diff = np.load("data/max_diff_sub.npy")
FA = np.load("data/FA_sub.npy")
size = np.load("data/dim_sub.npy")

x_dim = size[0]  # x dimension of subset
y_dim = size[1]  # y dimension of subset
z_dim = size[2]  # z dimension of subset

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
    dist = np.array([dist_lat(i, j) if i != j else np.float('infinity') for j in range(N_points)])
    return heapq.nsmallest(k, range(len(dist)), dist.take)


# computing nearest neighbors
print("Computing nearest neighbors...")
nn = set()
for i in range(N_points):
    print("Computing nearest neighbors for {} of {}".format(i,N_points))
    for j in wm_neighbors(i, k_neighbors):
        if i < j:
            nn.add((i, j))
        else:
            nn.add((j, i))

nn = list(nn)
N_neighbors = len(nn)

nn_to_index = {}
for i,v in enumerate(nn):
    nn_to_index[v] = i

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
    j_shape = 1 - np.abs(np.dot(vi, vj) / (sc.distance.norm(vi) * sc.distance.norm(vj)))
    j_proximity = 1 / k_neighbors * (nn_dist[nn_index] / (2 * d_avg))

    return j_shape * j_proximity

print("Computing Jij for all neighbors...")
nn_jij = np.array([j_cost(nn_index) for nn_index in range(N_neighbors)])

# initiate Cij and S
print("Initiating Cij and S...")
Cij = np.array([0 for i in nn])  # probability of finding sites i and j in the same cluster
S = np.ones(N_points) # Initialize S to ones

print("Starting Monte Carlo for t_superp = {}...".format(t_superp))
t_index = 0  # keep track of the burned-in samples
for t_i in range(t_iter):  # given iterations per temperature
    print("It. {}/{} \t Started Iteration...".format(t_i+1, t_iter))
    SS = [[] for i in range(q)]  # initialize SS

    G = nx.Graph()  # Initialize graph where we will store "frozen" bonds
    for i in range(N_points):
        G.add_node(i)

    for nn_index, (i, j) in enumerate(nn):  # nearest_neighbors has te be calculated in advance
        pfij = (1 - np.exp(-nn_jij[nn_index] / t_superp)) if S[i] == S[j] else 0  # page 9 of the paper
        if np.random.uniform(0, 1) < pfij:
            G.add_edge(i, j)

    subgraphs = list(nx.connected_component_subgraphs(G))  # find SW-clusters
    print("It. {}/{} \t {} subgraphs".format(t_i + 1, t_iter, len(subgraphs)))
    for graph in subgraphs:
        new_q = np.random.randint(1, q+1)
        for node in graph.nodes():
            SS[new_q-1].append(node)
            S[node] = new_q

    if t_index >= t_burn_in:
        for i in range(q):
            print("It. {}/{} \t Cij {}/{} \t Size: {}".format(t_i + 1, t_iter, i+1, q, len(SS[i])))
            for vi in SS[i]:
                for vj in SS[i]:
                    if vj > vi:
                        index = nn_to_index.get((vi,vj))
                        if index is not None:
                            Cij[index] += 1

    t_index += 1

print("Computing estimated probabilities...")
# average and obtain estimated probabilities
Cij = [i / (t_iter - t_burn_in) for i in Cij]

print("Computing spin-spin correlation...")
# calculate spin-spin correlation function Gij, (11) in the paper
Gij = [((q-1)*i+1)/q for i in Cij]

# initialize graph where we are going to construct our final clustering
print("Construct final graph and calculate clustering...")
G = nx.Graph()

for i in range(N_points):
    G.add_node(i)

for nn_index, g in enumerate(Gij):
    (i, j) = nn[nn_index]
    if g > Gij_threshold:
        G.add_edge(i, j)

Gij_current = [0 for i in range(N_points)]
best_neighbour = [0 for i in range(N_points)]
for nn_index, (vi, vj) in enumerate(nn): # capture points lying in the periphery
    if Gij[nn_index] > Gij_current[vi]:
        Gij_current[vi] = Gij[nn_index]
        best_neighbour[vi] = vj
    if Gij[nn_index] > Gij_current[vj]:
        Gij_current[vj] = Gij[nn_index]
        best_neighbour[vj] = vi

for vi, vj in enumerate(best_neighbour):
    G.add_edge(vi, vj)

# return final clustering
print("Formatting output...")
clusters = np.empty(N_points)
cluster_id = 1
for graph in list(nx.connected_component_subgraphs(G)):
    for node in graph.nodes():
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

id = '{:%d%m%y%H%M}'.format(datetime.now())
f = open('results/clustering_' + id + '.pkl', 'wb')
pickle.dump(results, f)
f.close()

end = time.time()

print("Finished in {} seconds!".format(end - start))
print("Exported with id {}".format(id))
