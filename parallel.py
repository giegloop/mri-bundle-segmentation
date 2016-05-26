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

#BLATT variables
q = 10  # num. of pot spin variables
k_neighbors = 25  # number of nearest neighbors


#MONTE CARLO Variables
t_iter_per_temp = 250  # num. of iterations per temperature
t_ini = 0.0001
t_end = 0.2
t_num_it = 50

########################################################################################################################

start = time.time()
num_cores = multiprocessing.cpu_count()

#Loading the subset and the directions
d = np.load('/home/antonio/Scrivania/MRI viktor/data/subset.npy')
bvecs = np.genfromtxt('/home/antonio/Scrivania/MRI viktor/data/bvecs', dtype='str')

# initialization of directions
bvecs_x = [float(i) for i in bvecs[:, 0]]
bvecs_y = [float(i) for i in bvecs[:, 1]]
bvecs_z = [float(i) for i in bvecs[:, 2]]
#bvecs = np.column_stack((bvecs_x, bvecs_y, bvecs_z))

#dimension of the data set
x_dim = len(d[:,0,0,0])
y_dim = len(d[0,:,0,0])
z_dim = len(d[0,0,:,0])

N_points = x_dim*y_dim*z_dim

# create array of all possible (x,y,z) within subset,
xyz = list(itertools.product(*[list(range(0, x_dim)), list(range(0, y_dim)), list(range(0, z_dim))]))

# creation of the data array (not a 4D tensor anymore) 
data = np.zeros((N_points, len(d[0,0,0,:])))

counter =0
for i in range(x_dim):
    for j in range(y_dim):
        for k in range(z_dim):
            data[counter, :] = d[i,j,k,:]
            counter+=1

# distance between two points in the lattice
def dist_lat(p1, p2):
    return sc.distance.euclidean(p1, p2)

# returns k-nearest neighbors for each point in the Lattice 
def wm_neighbors(i, k):
    dist = np.array([dist_lat(i, j) if i != j else np.float('infinity') for j in range(N_points)])
    return heapq.nsmallest(k, range(len(dist)), dist.take)
        
def compute_nearest_neighbors(i):
    nn=set()
    print("Computing nearest neighbors for {} of {}".format(i,N_points))
    for j in wm_neighbors(i, k_neighbors):
        if i < j:
            nn.add((i, j))
        else:
            nn.add((j, i))
    return nn

# computing nearest neighbors with parallel computation
print("Computing nearest neighbors...")
nn = set()
n = Parallel(n_jobs=num_cores)(delayed(compute_nearest_neighbors)(i) for i in range(N_points))
nn = set()

print("Merging the sets..")
for i in range(N_points):
    nn=nn.union(n[i])
    
nn = list(nn)
N_neighbors = len(nn)

# compute average distance
print("Computing (average) distances between neighbors...")
nn_dist = np.array([dist_lat(i, j) for (i, j) in nn])
d_avg = np.mean(nn_dist)
dSq_avg = np.mean(pow(nn_dist, 2))

#difference of the signal, ignoring the tensor model
def j_cost_not_normalized(nn_index):
    (i, j) = nn[nn_index]
    xi = np.multiply(bvecs_x, data[i, :])
    yi = np.multiply(bvecs_y, data[i, :])
    zi = np.multiply(bvecs_z, data[i, :])
    xj = np.multiply(bvecs_x, data[j, :])
    yj = np.multiply(bvecs_y, data[j, :])
    zj = np.multiply(bvecs_z, data[j, :])
    cost = np.sum(np.sqrt((xi-xj)**2+(yi-yj)**2+(zi-zj)**2))         
    return cost

print("Computing Jij for all neighbors...")
signal_difference = np.array([j_cost_not_normalized(nn_index) for nn_index in range(N_neighbors)])
mean_signal_difference = np.mean(signal_difference)
nn_jij = (float(1)/k_neighbors)*np.exp(-(signal_difference**2)/(2*mean_signal_difference**2))

T_expected =k_neighbors*np.mean(nn_jij)/(4*np.log(1+np.sqrt(10)))

t_trans = (1 / (4 * np.log(1 + np.sqrt(q)))) * np.exp(-dSq_avg / 2 * pow(d_avg, 2))  # page 14 of the paper


print("Start Monte Carlo with t_start = {}, t_end = {}, steps = {}...".format(t_ini, t_end, t_num_it))

mag_arr = []  # array with average magnetation of each time
mag_sq_arr = []  # array with average squared magnetation of each time
t_arr = np.arange(t_ini, t_end, (t_end - t_ini) / t_num_it)  # range of times to loop over

#Define a function for doing parallel computation
def do_MC_iteration(l,t):
   G = nx.Graph()  # Initialize graph where we will store "frozen" bonds
   for h in range(N_points):
       G.add_node(h)

   for nn_index, (i, j) in enumerate(nn):  # nearest_neighbors has te be calculated in advance
       pfij = (1 - np.exp(-nn_jij[nn_index] / t)) if S[i] == S[j] else 0  # page 9 of the paper
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

   return (q * N_max - N_points) / ((q - 1) * N_points)  # (4) in paper
    #END OF SWMC FUNCTION
        
        
#START THE SEARCH FOR GOOD TEMPERATURES
for ti,t in enumerate(t_arr):  # for each temperature
    print("Computing Thermodynamic quantities for T={}".format(t))
    S = np.ones(N_points)  # Initialize S to ones
    mag = 0
    magSq = 0
    mag = Parallel(n_jobs=num_cores)(delayed(do_MC_iteration)(i,t) for i in range(t_iter_per_temp))
    mag_arr.append(np.mean(mag) / t_iter_per_temp)
    mag_sq_arr.append(np.mean(np.power(mag,2)) / t_iter_per_temp)

mag_arr = np.array(mag_arr)
mag_sq_arr = np.array(mag_sq_arr)

# create susceptibility array
suscept_arr = (N_points / t_arr) * (mag_sq_arr - pow(mag_arr, 2))

# write results to file
results = {
    'x_dim': x_dim,
    'y_dim': y_dim,
    'z_dim': z_dim,
    'q': q,
    't_iter_per_temp': t_iter_per_temp,
    't_per_min': t_ini,
    't_per_max': t_end,
    't_num_it': t_num_it,
    'k_neighbors': k_neighbors,
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
