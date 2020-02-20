import logging
import numpy as np
import graph_tool as gt
import graph_tool.topology
import graph_tool.generation 
from src import utils as utils
from src import graph_utils as gu
import h5py
import skfmm
import numpy.ma as ma

def _sort_array(a):
  assert(a.ndim == 2)
  a_ = a*1
  for _ in range(a.shape[1], 0, -1):
    ind = np.argsort(a_[:,_-1], kind='mergesort')
    a_ = a_[ind,:]
  return a_

# Modify graph for computing distance along path.
def compress_edges_along_path(gtG, path, path_edge_wt, wt_ep_name='weight',
  edge_wt=None):
  gtG_ = gt.Graph(gtG)
  assert((wt_ep_name in gtG_.ep.keys()) ^ (edge_wt is not None))
  
  if wt_ep_name not in gtG_.ep.keys():
    wt = gtG_.new_edge_property('float')
    wt.a[...] = edge_wt
    gtG_.ep[wt_ep_name] = wt
  else:
    wt = gtG_.ep[wt_ep_name]
  for i in range(len(path)-1):
    wt[gtG_.edge(path[i], path[i+1])] = path_edge_wt
  return gtG_

# # Compute shortest path from all nodes to or from all source nodes
# def get_distance_along_path(gtG, path, path_edge_wt, wt_ep_name='weight',
#   edge_wt=None):
#   gtG_ = compress_edges_along_path(gtG, path, path_edge_wt, wt_ep_name, edge_wt)
#   wt = gtG_.ep[wt_ep_name]
#   v = gtG_.vertex(int(path[-1]))
#   dist = gt.topology.shortest_distance(gt.GraphView(gtG_, reversed=True), 
#     source=v, target=None, weights=wt)
#   dist = np.array(dist.get_array())
#   return dist

def _generate_graph_any_4(traversable, step_size, center=True):
  offset = [0, 0]
  if center:
    offset = [np.mod(traversable.shape[1], step_size)/2, 
      np.mod(traversable.shape[0], step_size)/2]
  logging.error(offset)
  
  t = traversable[offset[1]::step_size, offset[0]::step_size] 
  y, x = np.where(t)

  nodes = np.array([x,y]).T
  nodes = _sort_array(nodes)
  # ind = np.argsort(nodes[:,1], kind='mergesort')
  # nodes = nodes[ind, :]
  # ind = np.argsort(nodes[:,0], kind='mergesort')
  # nodes = nodes[ind,:]
  
  node_ids = -1*np.ones(t.shape, dtype=np.int32)
  aa = np.ravel_multi_index((nodes[:,1], nodes[:,0]), node_ids.shape)
  node_ids.ravel()[aa] = np.arange(aa.shape[0])
  k = nodes.shape[0]

  y_cumsum = np.cumsum(traversable, axis=0)[offset[1]::step_size, offset[0]::step_size]
  conn_yp = np.zeros(y_cumsum.shape, dtype=np.bool)
  conn_yp[:-1,:] = np.logical_and((y_cumsum[1:,:] - y_cumsum[:-1,:] == step_size), t[:-1,:])
  conn_yn = np.zeros(y_cumsum.shape, dtype=np.bool)
  conn_yn[1:,:] = np.logical_and((y_cumsum[1:,:] - y_cumsum[:-1,:] == step_size), t[:-1,:])
  
  edges_yp = np.concatenate((node_ids[:-1,:][conn_yp[:-1,:]][:,np.newaxis], 
                             node_ids[1:,:][conn_yp[:-1,:]][:,np.newaxis]), axis=1)
  assert(np.all(edges_yp >= 0))
  edges_yp[:,0] += 1*k 
  edges_yp[:,1] += 1*k 
  
  edges_yn = np.concatenate((node_ids[1:,:][conn_yn[1:,:]][:,np.newaxis], 
                             node_ids[:-1,:][conn_yn[1:,:]][:,np.newaxis]), axis=1)
  assert(np.all(edges_yn >= 0))
  edges_yn[:,0] += 3*k 
  edges_yn[:,1] += 3*k 
  
  x_cumsum = np.cumsum(traversable, axis=1)[offset[1]::step_size, offset[0]::step_size]
  conn_xp = np.zeros(x_cumsum.shape, dtype=np.bool)
  conn_xp[:,:-1] = np.logical_and((x_cumsum[:,1:] - x_cumsum[:,:-1] == step_size), t[:,:-1])
  conn_xn = np.zeros(x_cumsum.shape, dtype=np.bool)
  conn_xn[:,1:] = np.logical_and((x_cumsum[:,1:] - x_cumsum[:,:-1] == step_size), t[:,:-1])
  
  edges_xp = np.concatenate((node_ids[:,:-1][conn_xp[:,:-1]][:,np.newaxis], 
                             node_ids[:,1:][conn_xp[:,:-1]][:,np.newaxis]), axis=1)
  assert(np.all(edges_xp >= 0))
  edges_xp[:,0] += 0 
  edges_xp[:,1] += 0 
  
  edges_xn = np.concatenate((node_ids[:,1:][conn_xn[:,1:]][:,np.newaxis], 
                             node_ids[:,:-1][conn_xn[:,1:]][:,np.newaxis]), axis=1)
  assert(np.all(edges_xn >= 0))
  edges_xn[:,0] += 2*k 
  edges_xn[:,1] += 2*k 
  
  K = np.arange(nodes.shape[0])[:,np.newaxis]
  edges_tp = []; edges_tn = []
  for i in range(4):
    edges_tp.append(np.concatenate((K + i*k, K + np.mod(i+1,4)*k), axis=1))
    edges_tn.append(np.concatenate((K + i*k, K + np.mod(i-1,4)*k), axis=1))
  edges_tp = np.concatenate(edges_tp, axis=0)
  edges_tn = np.concatenate(edges_tn, axis=0)
  
  # Add rotational nodes
  nodes_theta_all = []
  for i in range(4):
    nodes_theta_all.append(
      np.concatenate((nodes*1, i*np.ones((nodes.shape[0],1), dtype=nodes.dtype)), axis=1))
  nodes = np.concatenate(nodes_theta_all, axis=0)
  edges_self = np.reshape(np.arange(nodes.shape[0], dtype=np.int64), [-1,1])
  edges_self = np.concatenate((edges_self, edges_self), axis=1)

  edges = [edges_xp, edges_xn, edges_yp, edges_yn, edges_tp, edges_tn, edges_self]
  for i in range(len(edges)):
    edges[i] = _sort_array(edges[i])
  
  nodes[:,0] = nodes[:,0]*step_size + offset[0]
  nodes[:,1] = nodes[:,1]*step_size + offset[1]
  return edges, nodes, offset
 
def _generate_graph_1_4(traversable):
  y, x = np.where(traversable)
  nodes = np.array([x,y]).T
  nodes = _sort_array(nodes)
  k = nodes.shape[0]

  # +y-edges
  conn = np.all(nodes[1:,:] - nodes[:-1,:] == np.array([[0,1]]), axis=1)
  first = np.where(conn)[0]
  edges_yp = np.array([first + 1*k, first+1 + 1*k]).T
  edges_yn = np.array([first + 1 + 3*k, first + 3*k]).T

  # +x-edges
  ind1 = np.argsort(nodes[:,0], kind='mergesort')
  n = nodes[ind1,:]
  ind2 = np.argsort(nodes[:,1], kind='mergesort')
  n = nodes[ind2,:]
  ind = ind1[ind2]
  assert(np.all(n == nodes[ind,:]))

  conn = np.all(n[1:,:] - n[:-1,:] == np.array([[1,0]]), axis=1)
  first = np.where(conn)[0]
  edges_xp = np.array([ind[first] + 0*k, ind[first+1] + 0*k]).T
  edges_xn = np.array([ind[first+1] + 2*k, ind[first] + 2*k]).T
  
  # Add rotation edges.
  K = np.arange(nodes.shape[0])[:,np.newaxis]
  edges_tp = []; edges_tn = []
  for i in range(4):
    edges_tp.append(np.concatenate((K + i*k, K + np.mod(i+1,4)*k), axis=1))
    edges_tn.append(np.concatenate((K + i*k, K + np.mod(i-1,4)*k), axis=1))
  edges_tp = np.concatenate(edges_tp, axis=0)
  edges_tn = np.concatenate(edges_tn, axis=0)
  
  # Add rotational nodes
  nodes_theta_all = []
  for i in range(4):
    nodes_theta_all.append(
      np.concatenate((nodes*1, i*np.ones((nodes.shape[0],1), dtype=nodes.dtype)), axis=1))
  nodes = np.concatenate(nodes_theta_all, axis=0)
  edges_self = np.reshape(np.arange(nodes.shape[0], dtype=np.int64), [-1,1])
  edges_self = np.concatenate((edges_self, edges_self), axis=1)

  edges = [edges_xp, edges_xn, edges_yp, edges_yn, edges_tp, edges_tn, edges_self]
  for i in range(len(edges)):
    edges[i] = _sort_array(edges[i])
  
  return edges, nodes
   
class NullGraph():
  """This Class computes a set of nodes in the free space and does not connect
  them in any way. Dummy class to set up data for training the mapper with
  greater viewpoint variability."""
  def __init__(self, traversable, force=False, step_size=1, seed=0):
    """Given a traversable map, computes an undirected grid graph on the
    space."""
    tt = utils.Timer()
    tt.tic()
    # Canonicalize the nodes.
    assert(np.sum(traversable) < 0.25*traversable.size or force), \
      'Doesn"t look like a sparse graph {:d} / {:d}'.format(np.sum(traversable), traversable.size)

    # Sample points in the free space.
    rng = np.random.RandomState(seed)
    # Compute number of samples such that number of points are the same between
    # the uniformly smapled graph and this.
    # Times 4 for orinetations.
    num_renders = int((np.sum(traversable)*4.) / (step_size*step_size*1.)) 
    ys, xs = np.where(traversable)
    ind = rng.choice(ys.shape[0], size=num_renders)

    perturbs = np.zeros((num_renders, 4), dtype=np.float32)
    perturbs[:,0] = xs[ind] + (rng.rand(num_renders)-0.5)
    perturbs[:,1] = ys[ind] + (rng.rand(num_renders)-0.5)
    perturbs[:,2] = rng.rand(num_renders)*4. # theta
    
    nodes = perturbs[:,:3]*1.
    offset = [0., 0.]
    
    g = gt.Graph(directed=True)
    g.add_vertex(n=nodes.shape[0])
    g.ep['action'] = g.new_edge_property('int')
    
    self.graph = g
    self.graph_props = utils.Foo(offset=offset, step_size=step_size, seed=seed)
    self.nodes = nodes
    self.node_ids = None
    tt.toc(log_at=1, log_str='NullGraph __init__: ', type='calls')

  def get_salt_string(self):
    return 'null-stepsize{:d}-seed{:d}'.format(self.graph_props.step_size, self.graph_props.seed)

class OrientedGridGraph():
  def __init__(self, traversible, force=False, step_size=1):
    """Given a traversable map, computes an undirected grid graph on the
    space."""
    self.map={}
    self.map['traversible_cc']=traversible
    self.map['resolution']=5 #attention
    #self.goal_dist, self.goal_dist_mask = self._compute_distance_field(goal)
    #self.state=start_state[0]
    #self.traj=[start_state[0]]
    self.du=step_size #in cm attention
    self.step_size=step_size
    self.noise=0 # attention
    self.n_ori=12#self.env.n_ori
    self.dt=np.pi/self.n_ori
    self.angle_value=[0,2.0*np.pi/self.n_ori,-2.0*np.pi/self.n_ori,0]
    self.free_xy=None
    # Canonicalize the nodes.
    #assert(np.sum(traversable) < 0.25*traversable.size or force), \
    #  'Doesn"t look like a sparse graph {:d} / {:d}'.format(np.sum(traversable), traversable.size)
    '''
    edges, nodes, offset = _generate_graph_any_4(traversable, step_size=step_size, center=True)
    edges_xp, edges_xn, edges_yp, edges_yn, edges_tp, edges_tn, edges_self = edges
    # _edges, _nodes = _generate_graph_1_4(traversable)
    # _edges_xp, _edges_xn, _edges_yp, _edges_yn, _edges_tp, _edges_tn, _edges_self = _edges
    # print(np.allclose(nodes, _nodes))
    # for e, e_ in zip(edges, _edges): print(np.allclose(e, e_))

    g = gt.Graph(directed=True)
    g.add_vertex(n=nodes.shape[0])
    g.ep['action'] = g.new_edge_property('int')
    
    g.add_edge_list(edges_xp)
    g.add_edge_list(edges_xn)
    g.add_edge_list(edges_yp)
    g.add_edge_list(edges_yn)
    g.ep['action'].a[...] = 3
    
    g.add_edge_list(edges_tp)
    assert(np.sum(g.ep['action'].a == 0) == edges_tp.shape[0])
    g.ep['action'].a[g.ep['action'].a == 0] = 1
    
    g.add_edge_list(edges_tn)
    assert(np.sum(g.ep['action'].a == 0) == edges_tn.shape[0])
    g.ep['action'].a[g.ep['action'].a == 0] = 2
    
    g.add_edge_list(edges_self)
    assert(np.sum(g.ep['action'].a == 0) == edges_self.shape[0])
    g.ep['action'].a[g.ep['action'].a == 0] = 0
    
    # import pdb; pdb.set_trace(); 
    # for es, typ in zip(all_edges, [3, 3, 3, 3, 1, 2, 0]):
    #   for e in es:
    #     _ = g.edge(*e)
    #     g.ep['action'][_] = typ
    
    # Label and prune away empty clusters in graph.
    comps = gt.topology.label_components(g)
    
    # Code to lookup the node id from the image
    node_ids = -1*np.ones(traversable.shape, dtype=np.int32)
    k = nodes.shape[0]/4
    aa = np.ravel_multi_index((nodes[:k,1], nodes[:k,0]), node_ids.shape)
    node_ids.ravel()[aa] = np.arange(aa.shape[0])
    
    self.graph = g
    self.graph_props = utils.Foo(offset=offset, step_size=step_size)
    self.nodes = nodes
    self.node_ids = node_ids
    self.component = np.array(comps[0].get_array())
    self.component_counts = comps[1]*1
    
    # Warm up library
    dist, pred_map = gt.topology.shortest_distance(
      gt.GraphView(self.graph, reversed=True), source=0, target=None, 
      max_dist=4, pred_map=True)
    tt.toc(log_at=1, log_str='OrientedGridGraph __init__: ', type='calls')
    ''' 
  def get_salt_string(self):
    return 'ogrid-stepsize{:d}'.format(self.graph_props.step_size)

  def _get_neighbours_helper(self, c):
    neigh = self.graph.vertex(c).out_neighbours()
    neigh = [int(x) for x in neigh]
    neigh_edge = self.graph.vertex(c).out_edges()
    neigh_action = [self.graph.ep['action'][e] for e in neigh_edge]
    return neigh, neigh_edge, neigh_action
  
  def get_neighbours(self, node_ids):
    """Returns the feasible set of actions from the current node."""
    a = -1*np.ones((len(node_ids), 4), dtype=np.int32)
    for i, c in enumerate(node_ids):
      neigh, neigh_edge, neigh_action = self._get_neighbours_helper(c)
      for n, ea in zip(neigh, neigh_action):
        a[i,ea] = n
    return a 

  def setup_rejection_sampling(self, min_dist, max_dist, n_bins, trials,
    rejection_sampling_M, target_d='uniform'):
    bins = np.arange(n_bins+1)/(n_bins*1.)
    rng = np.random.RandomState(0)
    if target_d == 'uniform':
      target_distribution = np.zeros(n_bins); target_distribution[...] = 1./n_bins;
    sampling_distribution = gu.get_hardness_distribution(self.graph, max_dist, min_dist, rng,
      trials, bins, self.nodes, 4, 1)
    self.rejection_sampling_data = utils.Foo(distribution_bins=bins,
      target_distribution=target_distribution,
      sampling_distribution=sampling_distribution,
      rejection_sampling_M=rejection_sampling_M, n_bins=n_bins)

  def sample_random_path_rejection(self, rng, min_dist=100, max_dist=3200):
    start_node_ids_, end_node_ids, dists, pred_maps, paths, hardnesss, gt_dists = \
      gu.rng_next_goal_rejection_sampling(None, 1, self.graph, rng, max_dist,
        min_dist, max_dist*2,
        self.rejection_sampling_data.sampling_distribution,
        self.rejection_sampling_data.target_distribution, self.nodes, 4, 1,
        self.rejection_sampling_data.distribution_bins,
        self.rejection_sampling_data.rejection_sampling_M)
    return start_node_ids_[0], end_node_ids[0], dists[0], dists[0][start_node_ids_[0]]
  
  def sample_random_path_U(self, rng, num_waypoints=1, min_dist=100,
    max_dist=3200, path_length=60, start_id=None):
    """Samples a random trajectory by going straight for some distance and then
    turning left or right and repeating."""
    waypoint_ids = []
    all_paths = []
    
    ids = np.arange(self.nodes.shape[0])
    id1 = start_id if start_id is not None else rng.choice(ids)
    waypoint_ids.append(id1);
    for i in range(num_waypoints):
      id1 = waypoint_ids[-1]
      n = rng.randint(min_dist, max_dist)
      path = []; path.append(id1);
      for _ in range(n):
        id1 = self.get_neighbours([id1])[0][3]
        if id1 == -1: 
          break
        path.append(id1)
      # Take a random rotation step.
      id1 = path[-1]
      pick = rng.randint(1,3)
      id1 = self.get_neighbours([id1])[0][pick]
      path.append(id1)
      all_paths.append(path)
      waypoint_ids.append(id1)
    all_paths = [np.array(x)[:-1] for x in all_paths]
    all_paths = np.concatenate(all_paths)
    
    # Clean up loops in the path by computing a distance field with these paths
    # and then following the shortest path from start to goal.
    all_paths = self.clean_path(all_paths.tolist(), path_edge_wt=0.1, 
      path_length=path_length)
    return waypoint_ids, all_paths 

  def _sample_point_on_map(self, rng, free_xy=None):
    if self.free_xy is None:
      self.free_xy=np.array(np.nonzero(self.map['traversible_cc'])).transpose()
    #print('road shape',self.task.road.shape)
    start_id = rng.choice(self.free_xy.shape[0])
    start_raw = self.free_xy[start_id]#*1. <-- attention
    #print(start_raw)
    start = start_raw #+ (rng.rand(2,)-0.5) <-- attention
    #start[0], start[1] = self._map_to_point(*start)
    #print(start)
    #assert(self.task.road[start[0], start[1]])
    return start_id, start_raw, start


  def _compute_distance_field(self, goal):
    self.goal=goal
    t = self.map['traversible_cc']
    #print('masked',t.shape)
    masked_t = ma.masked_values(t*1, 0)
    #print('goal',goal,masked_t.shape)
    goal_x, goal_y = int(goal[0]),int(goal[1])#_map_to_point((goal[0]), (goal[1])) # point on map attention 
    masked_t[goal_x, goal_y] = 0
    # scaling as per the reward scaling. 
    dd = skfmm.distance(masked_t, 0.25)#dx=self.map['resolution']/self.du) #self.map['resolution'])  
    dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
    dd = ma.filled(dd, np.max(dd)+1)
    self.goal_dist, self.goal_dist_mask = dd, dd_mask


  def _take_step(self, state, u_list):
    """Actions are discrete [0 (stay in place), 1(turn left), 2(turn right),
    3(straight ahead)].
    """
    #print(self.task_params.step_size)
    out_states = []
    angle_value=self.angle_value
    batch_size = len(u_list)
    prob_random = 0# attentionself.task_params.noise_model.prob_random
    action_status = []
    boundary_limits=self.map['traversible_cc'].shape
    #print(boundary_limits)
    x,y,t=list(state[0])

    collision_reward=0
    du=self.step_size
    for i in range(batch_size):
      action = u_list[i]
      x_new, y_new, t_new = x, y , t
      if action == 3:
        x_new=(x+(np.cos(t)*du))
        y_new=(y+(np.sin(t)*du))
      else:
        t_new=t+angle_value[action]

      if (np.array([int(x_new),int(y_new)])<np.array(boundary_limits)).all() and (np.array([int(x_new),int(y_new)])>=np.array([0,0])).all() and self.map['traversible_cc'][int(x_new),int(y_new)]==True  :
        x,y,t=x_new,y_new,t_new
      else:
        collision_reward=-1
        break

    return np.array([x,y,t])


  def _virtual_steps(self, u_list,state):
    """Actions are discrete [0 (stay in place), 1(turn left), 2(turn right),
    3(straight ahead)].
    """
    #print(self.task_params.step_size)
    out_states = []
    angle_value=self.angle_value
    batch_size = len(u_list)
    prob_random = 0# attentionself.task_params.noise_model.prob_random
    action_status = []
    boundary_limits=self.map['traversible_cc'].shape
    #print(boundary_limits)
    print(state)
    x,y,t=list(state[0])
    cost_start = self.goal_dist[int(x), int(y)] #Actual distance in cm.

    collision_reward=0
    du=self.step_size
    for i in range(batch_size):
      action = u_list[i]
      x_new, y_new, t_new = x, y , t
      if action == 3:
        x_new=(x+(np.cos(t)*du))
        y_new=(y+(np.sin(t)*du))
      else:
        t_new=t+angle_value[action]

      if (np.array([int(x_new),int(y_new)])<np.array(boundary_limits)).all() and (np.array([int(x_new),int(y_new)])>=np.array([0,0])).all() and self.map['traversible_cc'][int(x_new),int(y_new)]==True  :
        x,y,t=x_new,y_new,t_new
      else:
        collision_reward=-1
        break

    cost_end = self.goal_dist[int(x), int(y)]
    dist = cost_end*1.
    reward_near_goal = 0.
    if dist < self.du*4:
      reward_near_goal = 1.
    costs = (cost_end - cost_start)

    reward = -costs + reward_near_goal + collision_reward
    return reward

  def search_actions(self,num_rots):
    action_list=[[3]]
    append_list_pos=[]
    append_list_neg=[]
    for i in range(num_rots):
      append_list_pos.append(1)
      append_list_neg.append(2)
      action_list.append(append_list_pos[:]+[3])
      action_list.append(append_list_neg[:]+[3])

    return action_list

  def find_best_action(self,state):
    num_rots = int(self.n_ori/2)
    action_list=self.search_actions(num_rots)
    best_list=action_list[0]
    best_reward=self._virtual_steps(best_list,state)
    #print(best_reward,)
    for a_list in action_list[1:]:
      rew = self._virtual_steps(a_list,state)
      #print(rew,)
      if rew > best_reward:
        best_list=a_list
        best_reward=rew
    print('best_rew',best_reward)

    #np.set_printoptions(threshold='nan')
    #np.set_printoptions(linewidth=280)
    #np.set_printoptions(precision=1)
    #print(self.goal_dist[int(x)-10:int(x)+10,int(y)-10:int(y)+10])
    return best_list[0]

  def find_best_path(self,start_state,goal_state):
    self._compute_distance_field(goal_state)
    action_set=[]
    path_points=[start_state]
    reward_near_goal=0
    curr_state=start_state
    
    while(reward_near_goal==0):
      act=self.find_best_action([curr_state])
      curr_state=self._take_step([curr_state],[act])
      action_set.append(act)
      x,y,t=list(curr_state)
      cost_end = self.goal_dist[int(x), int(y)]
      dist = cost_end*1.

      if dist < self.du*4:
        reward_near_goal = 1.
      else:
        path_points.append(curr_state)
    
    return action_set, path_points

  def sample_random_path_waypoints(self, rng, num_waypoints=1, min_dist=100,
    max_dist=3200, path_length=60, start_id=None):
    # Sample a random trajectory in space in terms of going along the shortest
    # path from one point to another to another for multiple points.
    waypoint_ids = []
    all_paths = []
    rng=np.random.RandomState(10)
    id1 = np.array(list(self._sample_point_on_map(rng)[2])+[0])
    waypoint_ids.append(id1);
    for i in range(num_waypoints):
      id1 = waypoint_ids[-1]
      # Compute distance from the last point.
      id2 = np.array(list(self._sample_point_on_map(rng)[2])+[0])
      print(id1,id2)
      # Compute the trajectory that connecs id2 to id1.
      actions,path=self.find_best_path(id1,id2)
      all_paths.append(path); waypoint_ids.append(id2);

    #all_paths = [np.array(x)[:-1] for x in all_paths]
    all_paths = np.concatenate(all_paths)
    
    return waypoint_ids, all_paths 

  def sample_random_path_waypoints_graph(self, rng, num_waypoints=1, min_dist=100,
    max_dist=3200, path_length=60, start_id=None):
    # Sample a random trajectory in space in terms of going along the shortest
    # path from one point to another to another for multiple points.
    waypoint_ids = []
    all_paths = []
    
    #ids = np.arange(self.nodes.shape[0])
    id1 = start_id if start_id is not None else 0
    waypoint_ids.append(id1);
    for i in range(num_waypoints):
      id1 = waypoint_ids[-1]
      # Compute distance from the last point.
      dist, pred_map = gt.topology.shortest_distance(
        gt.GraphView(self.graph, reversed=True), source=id1, target=None,
        max_dist=max_dist+2, pred_map=True)
      node_dist = np.array(dist.get_array())
      node_dist[node_dist >= max_dist+2] = 2+max_dist
      
      # Pick out a new node from the set of feasible nodes.
      ids = np.where(np.logical_and(node_dist >= min_dist, node_dist <= max_dist))[0]
      if ids.size == 0:
        return self.sample_random_path_waypoints(rng, num_waypoints, min_dist, max_dist, path_length)
      id2 = rng.choice(ids)

      # Compute the trajectory that connecs id2 to id1.
      path = gt.topology.shortest_path(gt.GraphView(self.graph, reversed=False), 
        source=id1, target=self.graph.vertex(id2))
      path = [int(x) for x in path[0]]
      all_paths.append(path); waypoint_ids.append(id2);
    all_paths = [np.array(x)[:-1] for x in all_paths]
    all_paths = np.concatenate(all_paths)
    
    # Clean up loops in the path by computing a distance field with these paths
    # and then following the shortest path from start to goal.
    all_paths = self.clean_path(all_paths.tolist(), path_edge_wt=0.1, 
      path_length=path_length)
    return waypoint_ids, all_paths 
  
  def clean_path(self, path, path_edge_wt, path_length):
    # Returns the distance from all nodes such that descending on the
    # distance will cause you to follow the trajectories.
    gtG_ = compress_edges_along_path(self.graph, path, path_edge_wt=0.1,
      wt_ep_name='weight', edge_wt=1.0)
    wt = gtG_.ep['weight']
    s = int(path[0]); i = 1;
    i = 1
    while path[-i] == s: 
      i = i + 1
    e = path[-i]
    s = gtG_.vertex(s)
    e = gtG_.vertex(e)
    clean_path = gt.topology.shortest_path(gt.GraphView(gtG_, reversed=False), 
      source=s, target=e, weights=wt)
    clean_path = [int(x) for x in clean_path[0]]
    # clean_path = np.array(clean_path)[-(1+path_length):]
    clean_path = np.array(clean_path)[:(path_length+1)]
    rest = np.ones(np.maximum(0,path_length+1-clean_path.shape[0]), dtype=np.int64)
    rest = clean_path[-1]*rest
    clean_path = np.concatenate((clean_path, rest), axis=0)
    return clean_path

  def get_trajectory_distance(self, path, path_edge_wt, max_dist=None):
    # Returns the distance from all nodes such that descending on the
    # distance will cause you to follow the trajectories.
    gtG_ = compress_edges_along_path(self.graph, path, path_edge_wt=0.1,
      wt_ep_name='weight', edge_wt=1.0)
    wt = gtG_.ep['weight']
    v = gtG_.vertex(int(path[-1]))
    dist = gt.topology.shortest_distance(gt.GraphView(gtG_, reversed=True), 
      source=v, target=None, weights=wt, max_dist=max_dist)
    dist = np.array(dist.get_array())
    trajectory_distance = dist
    return trajectory_distance

  def _get_shortest_distance(self, node, max_dist):
    gtG_ = self.graph
    dist = gt.topology.shortest_distance(
      gt.GraphView(gtG_, reversed=True), source=gtG_.vertex(int(node)),
      target=None, max_dist=max_dist)
    dist = np.array(dist.get_array())
    return dist
  
  def get_path_distance(self, path):
    # Returns the distance to the path.
    path_dist = gu.get_distance_node_list(self.graph, path, 'to')
    return path_dist

  def _get_all_neighbours(self):
    if not hasattr(self, 'all_neighbours'):
      all_nodes = np.arange(self.nodes.shape[0], dtype=np.int32)
      self.all_neighbours = self.get_neighbours(all_nodes)
    return self.all_neighbours
  
  def get_action_distance(self, dist_field, node):
    tt = self.get_neighbours([node])[0,:]
    d = dist_field[tt]*1
    if np.issubdtype(d[0], np.int):
      max_val = np.iinfo(d.dtype).max
    else:
      max_val = np.finfo(d.dtype).max
    d[tt == -1] = max_val
    return tt, d

  def get_action_distance_vec(self, dist_field, nodes):
    all_nn = self._get_all_neighbours()
    tt = all_nn[nodes,:]*1
    # tt = self.get_neighbours(nodes)
    d = dist_field[tt]*1
    if np.issubdtype(d[0,0], np.int):
      max_val = np.iinfo(d.dtype).max
    else:
      max_val = np.finfo(d.dtype).max
    d[tt == -1] = max_val
    return tt, d

  def distance_to_field(self, distance):
    # Return a distance image as per the distances in the dist field.
    num_nodes = self.nodes.shape[0]/4
    node_id_image = self.node_ids
    fields = []
    step_size = self.graph_props.step_size
    offset = self.graph_props.offset
    for i in range(4):
      d = distance[node_id_image+num_nodes*i]
      d = d[offset[1]::step_size, offset[0]::step_size, np.newaxis]
      fields.append(d)
    fields = np.concatenate(fields, axis=2)
    return fields

  def get_actions(self, path):
    # Given a path, returns the sequence of actions to follow the path.
    action = self.graph.ep['action']
    acts = []
    for i in range(len(path)-1):
      e = self.graph.edge(path[i], path[i+1])
      acts.append(action[e])
    return acts

  def sample_random_goal(self, rng, min_dist, max_dist):
    """Samples a random path of length between min_dist and max_dist, returns
    the starting point and the end point.
    """
    ids = np.arange(self.nodes.shape[0])
    id1 = rng.choice(ids); 
    # Compute distance from the last point.
    dist, pred_map = gt.topology.shortest_distance(
      gt.GraphView(self.graph, reversed=True), source=id1, target=None,
      max_dist=max_dist+2, pred_map=True)
    node_dist = np.array(dist.get_array())
    node_dist[node_dist >= max_dist+2] = 2+max_dist
    
    # Pick out a new node from the set of feasible nodes.
    ids = np.where(np.logical_and(node_dist >= min_dist, node_dist <= max_dist))[0]
    if ids.size == 0:
      return self.sample_random_goal(rng, min_dist, max_dist)
    id2 = rng.choice(ids)
    path = gt.topology.shortest_path(self.graph, 
      self.graph.vertex(id1), self.graph.vertex(id2), pred_map=pred_map)
    path = [int(x) for x in path[0]]
    return id1, id2, path

def test_oriented_grid_graph():
  traversable = np.ones((200, 200), dtype=np.bool)
  graph = OrientedGridGraph(traversable, force=True, step_size=2)
  rng = np.random.RandomState(0)
  waypoints, path = graph.sample_random_path_waypoints(rng, 4, min_dist=4, 
    max_dist=200, path_length=60)
  assert(len(path) == 61)
  assert(path.dtype == np.int64)
  traj_dist = graph.get_trajectory_distance(path, 0.1)
  path_dist = graph.get_path_distance(path)
  path_field = graph.distance_to_field(path_dist)
  assert(np.allclose(np.array(list(path_field.shape)), np.array([100, 100, 4])))
  a = graph.get_actions(path)
  print(a)
