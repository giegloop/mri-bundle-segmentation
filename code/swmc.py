import numpy as np
import networkx as nx # graph library

# Define some parameters
x_dim = 210
y_dim = 210
z_dim = 130
N_points = x_dim*y_dim*z_dim
q = 20
iter_perT = 100

d_avg # TODO average of the distance between pairs of points
dSq_avg # TODO average of the square distance between pairs of points

t_trans = (1/(4*np.log(1+np.sqrt(q)))) * np.exp(-dSq_avg/2*d_avg) # page 14 of the paper

t_range = np.arange(0.9*t_trans, 1.1*t_trans, 0.01) # i just wrote a random step size 0.01
t_range = t_range[::-1] # reverse array to go from max to min like Viktor showed

# TODO Find neighbors for each point (K-nearest, Voronoi neighbors ?)

mag_arr = np.empty(len(t_range))
magSq_arr = np.empty(len(t_range))

for t in range(t_range): # for each temperature

  S = np.ones(N_points) # Initialize S to ones
  mag = 0
  magSq = 0

  for i in range(iter_perT): # given iterations per temperature

    G=nx.Graph() # Initialize graph where we will store "frozen" bonds
    for i in range(N_points):
      G.add_node(i)

    for i in range(N_points): # assign "frozen" bonds for neighbors
      neighbors = nearest_neighbors[i] # nearest_neighbors has te be calculated in advance
      for j in neighbors:
        Jij = compute_cost(i, j) # TODO Antonio
        pfij = (1 - np.exp(-Jij/t_range[t])) if S[i]==S[j] else 0 # page 9 of the paper
        if np.random.uniform(0,1) < pfij:
          G.add_edge(i,j)
    
    subgraphs = list(nx.connected_component_subgraphs(G)) # find SW-clusters
    for graph in subgraphs:
      new_q = np.random.randint(1,20)
      for node in graph.nodes():
        S[node] = new_q

    N_max = 0 # find N_max and compute mag and magSq, page 5 of the paper
    for q_val in range(q):
      new_N_max = sum(S==q_val)
      if new_N_max > N_max:
        N_max = new_N_max

    new_mag = (q*N_max-N_points)/((q-1)*N_points)
    mag = mag + new_mag
    magSq = magSq + new_mag**2
  
  mag_arr[t] = mag / iter_perT
  magSq_arr = magSq / iter_perT

# "COMPLETED" until the beginning of the page 14 of the paper
