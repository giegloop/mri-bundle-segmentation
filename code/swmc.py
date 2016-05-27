"""
Svendsen-Wang Algorithm for both swmc and clustering
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
# TYPE #
########################################################################################################################

type = 'clustering' # either 'swmc' or 'clustering'

########################################################################################################################
# GENERAL #
########################################################################################################################

q = 20  # num. of pot spin variables
mc_iterations = 400   # num. of iterations per MC
mc_burn_in = 50  # number of burn-in samples for MC (must be < mc_iterations!)
k_neighbors = 10  # number of nearest neighbors
wm_threshold = 0.25 # threshold for white mass (FA > wm_threshold is considered white mass)

########################################################################################################################
# SWMC #
########################################################################################################################

t_ini = 0.001  # initial temperature (cannot be 0!)
t_end = 1.5  # final temperature (must be > t_ini!)
t_num_it = 200  # number of iterations between initial and final temperature

########################################################################################################################
# CLUSTERING #
########################################################################################################################

t_superp = 0.47  # temperature in superparamagnetic phase
Cij_threshold = 0.5  # threshold for "core" clusters, section 4.3.2 of the paper

########################################################################################################################

if mc_burn_in >= mc_iterations:
    raise Exception('The number of MC burn-in samples is higher or equal to the number of MC iterations. Change it!')
if t_ini >= t_end:
    raise Exception('The initial temperature is higher or equal then the final temperature. Change it!')

start = time.time()

# load number of cores for parallel processing
num_cores = multiprocessing.cpu_count()
print("Number of cores: {}".format(num_cores))

# load data with maximum diffusion and fractional anisotropy
max_diff = np.load("data/max_diff_sub.npy")
print(max_diff.shape)
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
    j_shape = np.abs(np.dot(vi, vj) / (sc.distance.norm(vi) * sc.distance.norm(vj)))

    return np.exp(-6 * pow(j_shape, 2))

print("Computing Jij for all neighbors...")
nn_jij = np.array([j_cost(nn_index) for nn_index in range(N_neighbors)])

if type == 'swmc':
    print("Start Monte Carlo with t_start = {}, t_end = {}, steps = {}...".format(t_ini, t_end, t_num_it))

    mag_arr = []  # array with average magnetation of each time
    mag_sq_arr = []  # array with average squared magnetation of each time
    t_arr = np.arange(t_ini, t_end, (t_end - t_ini) / t_num_it)  # range of times to loop over

    # function per temperature step for parallization purposes
    def temp_step(t_i, t):
        print("Time: {}".format(t))

        S = np.zeros(N_points)  # initialize S to 1's (cluster assignment variable)
        mag = 0  # initialize magnetization
        mag_sq = 0  # initialize squared magnetization

        # loop through MC iterations
        for mc_index in range(mc_iterations):
            print("\t Iteration: {}/{}, time {}, {}/{}".format(mc_index + 1, mc_iterations, t, t_i + 1, len(t_arr)))

            # initialize graph where we will store "frozen" bonds
            G = nx.Graph()
            for i in range(N_points):
                G.add_node(i)

            # loop over all neighbors and "freeze" edge with certain prob.
            for nn_index, (i, j) in enumerate(nn):
                pfij = (1 - np.exp(-nn_jij[nn_index] / t)) if S[i] == S[j] else 0  # page 9 of the paper
                if np.random.uniform(0, 1) < pfij:
                    G.add_edge(i, j)

            # find SW clusters and assign new random cluster
            subgraphs = list(nx.connected_component_subgraphs(G))
            print("\t {} subgraphs".format(len(subgraphs)))
            for graph in subgraphs:
                new_q = np.random.randint(0, q)
                for node in graph.nodes():
                    S[node] = new_q

            # only compute magnetization when # of iterations >= # of burn-in samples
            if mc_index >= mc_burn_in:
                N_max = np.max(np.array([sum(S == q_val) for q_val in range(q)]))  # compute size of largest cluster
                new_mag = (q * N_max - N_points) / ((q - 1) * N_points)  # compute magnetization, (4) in paper

                mag += new_mag  # sum magnetization for estimate
                mag_sq += pow(new_mag, 2)  # sum squared magnetization for estimate

        # return mean of magnetization and squared magnetization
        return mag / (mc_iterations - mc_burn_in), mag_sq / (mc_iterations - mc_burn_in)

    # compute magnetization in parallel
    mag_new_arr = Parallel(n_jobs=num_cores)(delayed(temp_step)(t_i, t) for t_i, t in enumerate(t_arr))

    # extract magnetization and squared magnetization from parallel array
    mag_arr = np.array([i[0] for i in mag_new_arr])
    mag_sq_arr = np.array([i[1] for i in mag_new_arr])

    # compute susceptibility
    suscept_arr = (N_points / t_arr) * (mag_sq_arr - pow(mag_arr, 2))

    # write results to file
    results = {
        'x_dim': x_dim,
        'y_dim': y_dim,
        'z_dim': z_dim,
        'q': q,
        't_iter_per_temp': mc_iterations,
        't_burn_in': mc_burn_in,
        't_per_min': t_ini,
        't_per_max': t_end,
        't_num_it': t_num_it,
        'k_neighbors': k_neighbors,
        'wm_threshold': wm_threshold,
        'N_points': N_points,
        't_arr': t_arr,
        'mag_arr': mag_arr,
        'mag_sq_arr': mag_sq_arr,
        'suscept_arr': suscept_arr
    }

    id = '{:%d%m%y%H%M}'.format(datetime.now())
    f = open('results/results_' + id + '.pkl', 'wb')
    pickle.dump(results, f)
    f.close()

    end = time.time()

    print("Finished in {} seconds!".format(end - start))
    print("Exported with id {}".format(id))

elif type == 'clustering':
    # initiate Cij and S
    print("Initiating Cij and S...")
    Cij = np.array([0 for i in nn])  # probability of finding sites i and j in the same cluster
    S = np.zeros(N_points)  # initialize S to 0's (cluster assignment variable)

    print("Starting Monte Carlo for t_superp = {}...".format(t_superp))
    for mc_index in range(mc_iterations):  # given iterations per temperature
        print("It. {}/{} \t Started Iteration...".format(mc_index + 1, mc_iterations))
        SS = [[] for i in range(q)]  # initialize SS (list for each cluster containing spins that belong to it)

        # initialize graph where we will store "frozen" bonds
        G = nx.Graph()
        for i in range(N_points):
            G.add_node(i)

        # loop over all neighbors and "freeze" edge with certain prob.
        for nn_index, (i, j) in enumerate(nn):
            pfij = (1 - np.exp(-nn_jij[nn_index] / t_superp)) if S[i] == S[j] else 0  # page 9 of the paper
            if np.random.uniform(0, 1) < pfij:
                G.add_edge(i, j)

        # find SW clusters and assign new random cluster
        subgraphs = list(nx.connected_component_subgraphs(G))
        print("It. {}/{} \t {} subgraphs".format(mc_index + 1, mc_iterations, len(subgraphs)))
        for graph in subgraphs:
            new_q = np.random.randint(0, q)
            for node in graph.nodes():
                SS[new_q].append(node)
                S[node] = new_q

        # only compute correlation when enough burn-in samples
        if mc_index >= mc_burn_in:
            for i in range(q):
                print("It. {}/{} \t Cij {}/{} \t Size: {}".format(mc_index + 1, mc_iterations, i + 1, q, len(SS[i])))
                combinations = list(set(itertools.combinations(SS[i], 2)).intersection(set(nn)))
                combinations_keys = [nn_to_index[(i, j)] for (i, j) in combinations]
                for nn_index in combinations_keys:
                    Cij[nn_index] += 1

    # average and obtain estimated probabilities
    print("Computing estimated probabilities...")
    Cij = [i / (mc_iterations - mc_burn_in) for i in Cij]

    # initialize graph where we are going to construct our final clustering
    print("Construct final graph and calculate clustering...")
    G = nx.Graph()

    # add nodes for every spin
    for i in range(N_points):
        G.add_node(i)

    # add edges for neighboring spins
    for nn_index, g in enumerate(Cij):
        (i, j) = nn[nn_index]
        if g > Cij_threshold:
            G.add_edge(i, j)

    # compute best neighbors
    Cij_current = [0 for i in range(N_points)]
    best_neighbour = [0 for i in range(N_points)]
    for nn_index, (vi, vj) in enumerate(nn):
        if Cij[nn_index] > Cij_current[vi]:
            Cij_current[vi] = Cij[nn_index]
            best_neighbour[vi] = vj
        if Cij[nn_index] > Cij_current[vj]:
            Cij_current[vj] = Cij[nn_index]
            best_neighbour[vj] = vi

    # add edge for every best neighbor
    for vi, vj in enumerate(best_neighbour):
        G.add_edge(vi, vj)

    # return final clustering
    print("Formatting output...")
    clusters = np.empty(N_points)
    cluster_id = 0
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
        't_iter_per_temp': mc_iterations,
        't_burn_in': mc_burn_in,
        'k_neighbors': k_neighbors,
        'wm_threshold': wm_threshold,
        'Cij_threshold': Cij_threshold,
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