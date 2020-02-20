import logging
import numpy as np
from src import utils
from env import factory 
from env import toy_landmark_env as tle 
from env import mp_env

def get_mapper_task_params(batch_size, num_samples, extent_samples,
  add_flips=False, mapper_noise=0., repeat_map_samples=1, test_f=None, 
  output_optimal_actions=False):
  """Task Parameters for mapper."""
  t = utils.Foo(batch_size=batch_size, num_samples=num_samples,
    extent_samples=extent_samples, add_flips=add_flips,
    mapper_noise=mapper_noise, repeat_map_samples=repeat_map_samples, 
    test_f=test_f, output_optimal_actions=output_optimal_actions)
  return t

def _get_relative_loc(target, ref, theta):
  rel_goal_orientation = np.mod(np.int32((theta - _theta)/(np.pi/2)), 4)
  r = np.sqrt(np.sum(np.square(target - ref), axis=1))
  t = np.arctan2(target[:,1] - ref[:,1], target[:,0] - ref[:,0])
  t = t-theta[:,0] + np.pi/2
  return np.expand_dims(r,axis=1), np.expand_dims(t, axis=1)

class MapperEnv():
  """Provides data for the follower leader style problem. The leader generates
  a target trajectory, and outputs the set of images seen in going from the
  starting location to the goal location. Follower has noisy actuators and has
  to be able to get to the goal location the picked out by the leader."""
  
  def __init__(self, env, task_params):
    self.task = utils.Foo()
    self.task.env = env
    self.task_params = task_params
    self.repeat = utils.Foo(iter=0, ids=None, id_samples=None)
    assert(task_params.output_optimal_actions == False)
  
  def gen_data(self, rng):
    """Generates data for the mapping problem."""
    env = self.task.env
    nodes = env.task.graph.nodes
    tp = self.task_params
    tp_ = env.task_params
    batch_size = tp.batch_size
    
    # Sample a location in space.
    if np.mod(self.repeat.iter, self.task_params.repeat_map_samples) == 0:
      ids = rng.choice(nodes.shape[0], size=(batch_size))
      id_samples = self._sample_mapping_nodes(ids, rng)
      self.repeat.ids, self.repeat.id_samples = ids, id_samples
    else:
      # Sample such that the mapping data is the same.
      ids = self.repeat.ids;
      id_samples = self._sample_mapping_nodes(ids, rng)
      num_samples = self.task_params.num_samples
      num_test = int(round(self.task_params.test_f*num_samples))
      num_train = num_samples - num_test
      id_samples[:,:num_train] = self.repeat.id_samples[:,:num_train]*1

    outputs = self._gen_mapping_data(id_samples, ids, rng)
    self.repeat.iter = np.mod(self.repeat.iter+1, self.task_params.repeat_map_samples)
    return outputs
  
  def _sample_mapping_nodes(self, ids, rng):
    # Sample some of these nodes.
    env = self.task.env
    nodes = env.task.graph.nodes
    tp = self.task_params
    tp_ = env.task_params
    batch_size = tp.batch_size
    
    # Figure out nodes that are within some neighborhood of the current
    # location (in cm).
    # dist = np.linalg.norm(nodes[np.newaxis, :,:2] - nodes[ids, np.newaxis, :2], axis=2) #L2 Distance
    dist = np.max(np.abs(nodes[np.newaxis, :,:2] - nodes[ids, np.newaxis, :2]), axis=2)   #Linfinity norm
    dist = dist*env.task.building.env.resolution
  
    id_samples = []
    for i in range(batch_size):
      # Have some arbitrary orientations.
      id_ = rng.choice(np.where(dist[i,:] < tp.extent_samples)[0], size=tp.num_samples) 
      id_samples.append(id_)
    id_samples = np.array(id_samples)
    return id_samples

  def _gen_mapping_data(self, id_samples, ids, rng=None):
    """Given id_samples and ids renders out the relevant images and computes
    the relative orientations etc."""
    env = self.task.env
    tp_ = env.task_params
    tp = self.task_params
    goal_dist, goal_theta, rel_goal_orientation = \
      env.get_relative_coordinates(id_samples, ids)
    goal_dist = goal_dist * env.task.building.env.resolution

    if tp.mapper_noise > 0:
      assert(rng is not None)
      # Add noise to the goal_dist.
      goal_dist = goal_dist * (1+(rng.rand(*goal_dist.shape)-0.5)*tp.mapper_noise)


    # Render these auxxiliary nodes.
    views = env.render_views(id_samples.ravel())
    views = np.reshape(views, 
      [id_samples.shape[0], id_samples.shape[1]] + list(views[0].shape[1:]))

    # Get map at the location in space.
    roads = {}
    ff = env.get_features(ids)
    for i in range(len(tp_.fovs)): 
      roads['roads_{:d}'.format(i)] = ff['roads_{:d}'.format(i)]

    # Populate outputs.
    outputs = {}
    outputs['views'] = views
    outputs.update(roads)
    outputs['dist'] = goal_dist
    outputs['theta'] = goal_theta 
    outputs['rel_orient'] = rel_goal_orientation 
    return outputs

  def _vis_data(self, outputs):
    # Visualize the data in outputs.
    import matplotlib.pyplot as plt, os
    batch_size = outputs['dist'].shape[0]
    num_samples = outputs['dist'].shape[1]
    colstr = 'rgbmck'
    colstr += colstr
    for i in range(batch_size):
      fig, _, axes = utils.subplot2(plt, (2, int(np.ceil((2.+num_samples)/2))), (5,5))
      ax = axes.pop()
      ax.imshow(outputs['roads_0'][i,:,:,0], origin='lower')
      
      ax = axes.pop()
      sz = 30
      x = outputs['dist'][i,:]*np.cos(outputs['theta'][i,:])
      y = outputs['dist'][i,:]*np.sin(outputs['theta'][i,:])
      ax.plot(x, y, 'k.'); ax.plot(0., 0., 'r.')
      dx = sz*np.cos(np.pi/2.-outputs['rel_orient'][i,:]*np.pi/2.)
      dy = sz*np.sin(np.pi/2.-outputs['rel_orient'][i,:]*np.pi/2.)
      for j in range(num_samples): ax.arrow(x[j], y[j], dx[j], dy[j])
      for j in range(num_samples): ax.text(x[j], y[j], '{:d}'.format(j))
      
      sz = 128/0.5*5./2.
      ax.imshow(outputs['roads_0'][i,:,:,0], origin='lower', extent=[-sz, sz, -sz, sz])
      
      for j in range(num_samples): 
        ax = axes.pop(); ax.imshow(outputs['views'][i,j,...].astype(np.uint8)); 
        ax.set_title('{:d}'.format(j)); ax.axis('off')

      out_file_name = os.path.join('tmp/mapper_vis/{:d}.jpg'.format(i))
      fig.savefig(out_file_name, bbox_inches='tight', transparent=True, pad_inches=0)
      plt.close(fig)

class MapperPlannerEnv(MapperEnv):
  def __init__(self, env, task_params):
    self.task = utils.Foo()
    self.task.env = env
    self.task_params = task_params
    self.repeat = utils.Foo(iter=0, ids=None, id_samples=None)
    env = self.task.env
    
  def _get_node_nn(self, center_id):
    """Returns a neighborhood of nodes around center_id."""
    env = self.task.env
    graph = self.task.env.task.graph
    nodes = env.task.graph.nodes
    tp = self.task_params
    tp_ = env.task_params
    batch_size = tp.batch_size
    nn_size = self.task.nn_size

    o = center_id / (nodes.shape[0]/4)
    ll = np.mod(center_id, nodes.shape[0]/4)
    node_ids_on_graph_pad = self.node_ids_on_graph_pads[o]
    
    _i, _j = np.where(node_ids_on_graph_pad == ll)
    goal_nn = node_ids_on_graph_pad[_i[0]-nn_size:_i[0]+nn_size+1,_j[0]-nn_size:_j[0]+nn_size+1]*1
    goal_nn = goal_nn[:,:,np.newaxis]
    goal_nn = np.concatenate([goal_nn, goal_nn, goal_nn, goal_nn], 2)
    for j in range(4): _ = goal_nn[:,:,j]; _[_ > -1] = _[_ > -1] + np.mod(o+j,4)*nodes.shape[0]/4
    # np.set_printoptions(linewidth=260, threshold=10000)
    # print(goal_nn[:,:,0])
    # print(graph.get_neighbours([center_id]))
    # print(center_id)
    # center_id = ll + 3*nodes.shape[0]/4
    return goal_nn

  def gen_data(self, rng):
    """Generates data for the mapping problem."""
    env = self.task.env
    graph = self.task.env.task.graph
    nodes = env.task.graph.nodes
    tp = self.task_params
    tp_ = env.task_params
    batch_size = tp.batch_size
    assert(self.task_params.repeat_map_samples == 1)
    
    # Sampling a node with orientation 0 (so that we don't have to rotate things here)
    ids = rng.choice(nodes.shape[0]/4, size=(batch_size)) 
    id_samples = self._sample_mapping_nodes(ids, rng)
    self.repeat.ids, self.repeat.id_samples = ids, id_samples
    outputs = self._gen_mapping_data(id_samples, ids, rng)
    self.repeat.iter = np.mod(self.repeat.iter+1, self.task_params.repeat_map_samples)
    
    node_ids_on_graph_pad = self.task.node_ids_on_graph_pads[0]
    nn_size = self.task.nn_size
    
    # For each node that we sampled, compute shortest path within a window.
    all_oas, all_goal_nn, all_goal_dists = [], [], []
    for i in range(batch_size):
      # Find inds[i] in node_ids_on_graph on graph and crop out part of the map.
      _i, _j = np.where(node_ids_on_graph_pad == ids[i])
      goal_nn = node_ids_on_graph_pad[_i[0]-nn_size:_i[0]+nn_size+1,_j[0]-nn_size:_j[0]+nn_size+1]
      goal_nn = goal_nn[:,:,np.newaxis]
      goal_nn = np.concatenate([goal_nn, goal_nn, goal_nn, goal_nn], 2)
      for j in range(4): 
        _ = goal_nn[:,:,j]; _[_ > -1] = _[_ > -1] + j*nodes.shape[0]/4
      goal_dists = graph._get_shortest_distance(ids[i], nn_size*4)
      # goal_dists = graph.get_path_distance([ids[i]])
      _ = goal_nn[goal_nn > -1]
      _n, _d = graph.get_action_distance_vec(goal_dists, _)
      _oa = _d == np.min(_d, 1)[:,np.newaxis]
      oas = np.zeros((goal_nn.shape[0], goal_nn.shape[1], goal_nn.shape[2], 4), dtype=np.bool)
      oas[goal_nn > -1, :] = _oa
      # oas = []
      # for j in range(4):
      #   _oas = np.zeros((goal_nn.shape[0], goal_nn.shape[1], 4), dtype=np.bool)
      #   _n, _d = graph.get_action_distance_vec(goal_dists, _ + j*nodes.shape[0]/4)
      #   _oa = _d == np.min(_d, 1)[:,np.newaxis]
      #   _oas[goal_nn > -1, :] = _oa
      #   oas.append(_oas)
      # oas = np.concatenate(oas, 2)
      all_oas.append(oas); 
      all_goal_nn.append(goal_nn);
      all_goal_dists.append(goal_dists)

    all_oas = np.array(all_oas) # bs x 33 x 33 x 4 x 4
    all_goal_nn = np.array(all_goal_nn) # bs x 33 x 33 x 4
    
    all_oas = all_oas[:,::-1,:,:,:]
    all_oas = np.transpose(all_oas, [0,2,1,3,4])
    all_oas = all_oas.astype(np.float32)
    all_goal_nn = all_goal_nn[:,::-1,:,:]
    all_goal_nn = np.transpose(all_goal_nn, [0,2,1,3])
    valid_mask = all_goal_nn[:,:,:,:1] > -1
    outputs['valid'] = valid_mask.astype(np.float32)
    outputs['gt_actions'] = all_oas 
    self.episode = utils.Foo(all_goal_dists=all_goal_dists, outputs=outputs, all_goal_nn=all_goal_nn)
    return outputs

  def decode_plan(self, action_volume, init_states=None, goal_dists=None,
    goal_nn=None, num_steps=None):
    # Given the action volume around the goal location, walks from some number
    # of start locations by following the maximum action and seeing the final
    # distance to the goal location.
    """Adds teacher_actions, teacher_xyt, teacher_dist, teacher_theta,
    teacher_rel_orient, teacher_views to the dictionary."""
    env = self.task.env
    tp = self.task_params
    tp_ = env.task_params
    batch_size = tp.batch_size
    act_ind = np.argmax(action_volume, 4)
    start = [20, 20, 0]
    all_states, all_actions = [], []
    for i in range(batch_size):
      state = np.array(start)*1
      actions, states = [], []
      for j in range(num_steps):
        if (state[0] < 0 or state[1] < 0 or state[0] > act_ind.shape[1]-1 or
          state[1] > act_ind.shape[2]-1): act = 0
        else: act = act_ind[i, state[0], state[1], state[2]]
        states.append(state*1)
        actions.append(act*1)
        if act == 0: None
        elif act == 1: state[2] = np.mod(state[2]+1, 4)
        elif act == 2: state[2] = np.mod(state[2]-1, 4)
        else:
          if state[2] == 0: state[0] = state[0]+1
          elif state[2] == 1: state[1] = state[1]-1
          elif state[2] == 2: state[0] = state[0]-1
          elif state[2] == 3: state[1] = state[1]+1
      all_actions.append(actions)
      all_states.append(states)
    all_actions = np.array(all_actions)
    all_states = np.array(all_states)
    
    _all_states = all_states
    _all_actions = all_actions
    _state_dist = np.linalg.norm(_all_states[:,:,:2]-np.array(start[:2]), axis=2)*8
    _state_theta = np.arctan2(_all_states[:,:,0]-start[0], _all_states[:,:,1]-start[1])
    _state_orientation = np.mod(-_all_states[:,:,2] + start[2], 4)
    
    # Debug to print distance along traversal.
    # ds = np.ones((batch_size,20), dtype=np.float32)*np.inf
    # for ii in range(batch_size):
    #   for i in range(20): 
    #     _ = all_states[ii,i,:]
    #     if goal_nn[_[0],_[1],_[2]] > 0:
    #       ds[ii,i] = goal_dists[ii][goal_nn[_[0],_[1],_[2]]]
    # print(ds.astype(np.int32))
    
    # states = init_states
    # all_actions, all_states = [], []
    # all_states.append(states)
    # for j in range(batch_size): 
    #   d_starts.append(goal_dists[j][states[j]])
    #   
    # for k in range(num_steps):
    #   acts = []
    #   for j in range(batch_size): 
    #     action_vol = act_ind[j,...]; 
    #     act = action_vol[goal_nn == states[j]]
    #     act = 0 if len(act) == 0 else act[0]
    #     acts.append(act)
    #   states = env.take_action(states, acts)
    #   all_states.append(states)
    #   all_actions.append(acts)
    # all_states = np.array(all_states).T
    # all_actions = np.array(all_actions).T
    # state_dist, state_theta, state_orientation = env.get_relative_coordinates(all_states, init_states)
    return _all_actions, _state_dist, _state_theta, _state_orientation
  
  def execute_actions_1(self, action_volume, init_states, goal_dists, goal_nn, num_steps):
    # Given the action volume around the goal location, walks from some number
    # of start locations by following the maximum action and seeing the final
    # distance to the goal location.
    env = self.task.env
    tp = self.task_params
    tp_ = env.task_params
    batch_size = tp.batch_size
    
    action_volume = np.argmax(action_volume, -1) 
    d_starts, d_ends = [], []
    state_thetas, state_dists = [], []
    all_states = []
    
    states = init_states
    all_states.append(states)
    for j in range(batch_size): 
      d_starts.append(goal_dists[j][states[j]])
      
    for k in range(num_steps):
      acts = []
      for j in range(batch_size): 
        action_vol = action_volume[j,...]; 
        act = action_vol[goal_nn == states[j]]
        act = 0 if len(act) == 0 else act[0]
        acts.append(act);
      states = env.take_action(states, acts)
      all_states.append(states)
    all_states = np.array(all_states).T
    # state_dist, state_theta, state_orientation = \
    #   env.get_relative_coordinates(all_states, self.repeat.ids)
    # state_dists.append(state_dist)
    # state_thetas.append(state_theta)

    d_ends = []
    for j in range(batch_size): 
      d_ends.append(goal_dists[j][states[j]])

    d_ends = np.array(d_ends)
    d_starts = np.array(d_starts)
    # state_dists = np.array(state_dists)*env.task.building.env.resolution
    # state_thetas = np.array(state_thetas)
    return d_starts, d_ends #, (state_dists, state_thetas) 

  def execute_actions(self, action_volume, min_dist=0, num_starts=20):
    # Given the action volume around the goal location, walks from some number
    # of start locations by following the maximum action and seeing the final
    # distance to the goal location.
    env = self.task.env
    graph = self.task.env.task.graph
    nodes = env.task.graph.nodes
    episode = self.episode
    tp = self.task_params
    tp_ = env.task_params
    batch_size = tp.batch_size
    nn_size = self.task.nn_size
 
    num_steps = nn_size*2
    max_dists = nn_size
    rng = np.random.RandomState(0)
    rng_env = np.random.RandomState(0)
    action_volume = np.argmax(action_volume, -1) 
    d_starts, d_ends, all_states, state_thetas, state_dists = [], [], [], [], []
    for i in range(num_starts):
      init_states = [];
      all_states = [];
      for j in range(batch_size):
        goal_dist = episode.all_goal_dists[j]
        starts = np.where(np.logical_and(goal_dist >= min_dist, goal_dist < max_dists))[0]
        if len(starts) == 0:
          starts = np.where(np.logical_and(goal_dist >= 0, goal_dist < max_dists))[0]
        init_state = rng.choice(starts)
        assert(init_state in episode.all_goal_nn[j])
        init_states.append(init_state)
      states = env.reset(rng_env, init_states=init_states, batch_size=None)
      all_states.append(states)
      d_start = []
      for j in range(batch_size): 
        d_start.append(episode.all_goal_dists[j][states[j]])

      for k in range(num_steps):
        acts = []
        for j in range(batch_size): 
          goal_nn = episode.all_goal_nn[j]; action_vol = action_volume[j,...]; 
          act = action_vol[goal_nn == states[j]]
          act = 0 if len(act) == 0 else act[0]
          acts.append(act);
        states = env.take_action(states, acts)
        all_states.append(states)
      all_states = np.array(all_states).T
      state_dist, state_theta, state_orientation = \
        env.get_relative_coordinates(all_states, self.repeat.ids)
      state_dists.append(state_dist)
      state_thetas.append(state_theta)

      # self.repeat.ids
      # Compute distance to goal at end of episode
      d_end = []
      for j in range(batch_size): 
        d_end.append(episode.all_goal_dists[j][states[j]])
      d_ends.append(d_end)
      d_starts.append(d_start)

    d_ends = np.array(d_ends)
    d_starts = np.array(d_starts)
    state_dists = np.array(state_dists)*env.task.building.env.resolution
    state_thetas = np.array(state_thetas)
    return d_starts, d_ends, (state_dists, state_thetas) 


class MapperPlannerEnvGrid(MapperEnv):
  def __init__(self, env, task_params):
    self.task = utils.Foo()
    self.task.env = env
    self.task_params = task_params
    self.repeat = utils.Foo(iter=0, ids=None, id_samples=None)
    env = self.task.env
    graph = self.task.env.task.graph
    nodes = env.task.graph.nodes
    
    offset, step_size = env.task.graph.graph_props.offset, env.task.graph.graph_props.step_size
    node_ids_on_graph = env.task.graph.node_ids[offset[1]::step_size, offset[0]::step_size]*1
    nn_size = self.task.nn_size = (env.task_params.fovs[0]-1)/2
    node_ids_on_graph_pad = np.pad(node_ids_on_graph, (nn_size, nn_size), 'constant', constant_values=-1)
    self.node_ids_on_graph_pads = [node_ids_on_graph_pad]
    for i in range(3):
      n = self.node_ids_on_graph_pads[-1]*1
      # n = n.T; n = n[:,::-1]
      n = n.T; n = n[::-1,:]
      self.node_ids_on_graph_pads.append(n)

  def _get_node_nn(self, center_id):
    """Returns a neighborhood of nodes around center_id."""
    env = self.task.env
    graph = self.task.env.task.graph
    nodes = env.task.graph.nodes
    tp = self.task_params
    tp_ = env.task_params
    batch_size = tp.batch_size
    nn_size = self.task.nn_size

    o = center_id / (nodes.shape[0]/4)
    ll = np.mod(center_id, nodes.shape[0]/4)
    node_ids_on_graph_pad = self.node_ids_on_graph_pads[o]
    
    _i, _j = np.where(node_ids_on_graph_pad == ll)
    goal_nn = node_ids_on_graph_pad[_i[0]-nn_size:_i[0]+nn_size+1,_j[0]-nn_size:_j[0]+nn_size+1]*1
    goal_nn = goal_nn[:,:,np.newaxis]
    goal_nn = np.concatenate([goal_nn, goal_nn, goal_nn, goal_nn], 2)
    for j in range(4): _ = goal_nn[:,:,j]; _[_ > -1] = _[_ > -1] + np.mod(o+j,4)*nodes.shape[0]/4
    # np.set_printoptions(linewidth=260, threshold=10000)
    # print(goal_nn[:,:,0])
    # print(graph.get_neighbours([center_id]))
    # print(center_id)
    # center_id = ll + 3*nodes.shape[0]/4
    return goal_nn

  def gen_data(self, rng):
    """Generates data for the mapping problem."""
    env = self.task.env
    graph = self.task.env.task.graph
    nodes = env.task.graph.nodes
    tp = self.task_params
    tp_ = env.task_params
    batch_size = tp.batch_size
    assert(self.task_params.repeat_map_samples == 1)
    
    # Sampling a node with orientation 0 (so that we don't have to rotate things here)
    ids = rng.choice(nodes.shape[0]/4, size=(batch_size)) 
    id_samples = self._sample_mapping_nodes(ids, rng)
    self.repeat.ids, self.repeat.id_samples = ids, id_samples
    outputs = self._gen_mapping_data(id_samples, ids, rng)
    self.repeat.iter = np.mod(self.repeat.iter+1, self.task_params.repeat_map_samples)
    
    node_ids_on_graph_pad = self.task.node_ids_on_graph_pads[0]
    nn_size = self.task.nn_size
    
    # For each node that we sampled, compute shortest path within a window.
    all_oas, all_goal_nn, all_goal_dists = [], [], []
    for i in range(batch_size):
      # Find inds[i] in node_ids_on_graph on graph and crop out part of the map.
      _i, _j = np.where(node_ids_on_graph_pad == ids[i])
      goal_nn = node_ids_on_graph_pad[_i[0]-nn_size:_i[0]+nn_size+1,_j[0]-nn_size:_j[0]+nn_size+1]
      goal_nn = goal_nn[:,:,np.newaxis]
      goal_nn = np.concatenate([goal_nn, goal_nn, goal_nn, goal_nn], 2)
      for j in range(4): 
        _ = goal_nn[:,:,j]; _[_ > -1] = _[_ > -1] + j*nodes.shape[0]/4
      goal_dists = graph._get_shortest_distance(ids[i], nn_size*4)
      # goal_dists = graph.get_path_distance([ids[i]])
      _ = goal_nn[goal_nn > -1]
      _n, _d = graph.get_action_distance_vec(goal_dists, _)
      _oa = _d == np.min(_d, 1)[:,np.newaxis]
      oas = np.zeros((goal_nn.shape[0], goal_nn.shape[1], goal_nn.shape[2], 4), dtype=np.bool)
      oas[goal_nn > -1, :] = _oa
      # oas = []
      # for j in range(4):
      #   _oas = np.zeros((goal_nn.shape[0], goal_nn.shape[1], 4), dtype=np.bool)
      #   _n, _d = graph.get_action_distance_vec(goal_dists, _ + j*nodes.shape[0]/4)
      #   _oa = _d == np.min(_d, 1)[:,np.newaxis]
      #   _oas[goal_nn > -1, :] = _oa
      #   oas.append(_oas)
      # oas = np.concatenate(oas, 2)
      all_oas.append(oas); 
      all_goal_nn.append(goal_nn);
      all_goal_dists.append(goal_dists)

    all_oas = np.array(all_oas) # bs x 33 x 33 x 4 x 4
    all_goal_nn = np.array(all_goal_nn) # bs x 33 x 33 x 4
    
    all_oas = all_oas[:,::-1,:,:,:]
    all_oas = np.transpose(all_oas, [0,2,1,3,4])
    all_oas = all_oas.astype(np.float32)
    all_goal_nn = all_goal_nn[:,::-1,:,:]
    all_goal_nn = np.transpose(all_goal_nn, [0,2,1,3])
    valid_mask = all_goal_nn[:,:,:,:1] > -1
    outputs['valid'] = valid_mask.astype(np.float32)
    outputs['gt_actions'] = all_oas 
    self.episode = utils.Foo(all_goal_dists=all_goal_dists, outputs=outputs, all_goal_nn=all_goal_nn)
    return outputs

  def decode_plan(self, action_volume, init_states=None, goal_dists=None,
    goal_nn=None, num_steps=None):
    # Given the action volume around the goal location, walks from some number
    # of start locations by following the maximum action and seeing the final
    # distance to the goal location.
    """Adds teacher_actions, teacher_xyt, teacher_dist, teacher_theta,
    teacher_rel_orient, teacher_views to the dictionary."""
    env = self.task.env
    tp = self.task_params
    tp_ = env.task_params
    batch_size = tp.batch_size
    act_ind = np.argmax(action_volume, 4)
    start = [20, 20, 0]
    all_states, all_actions = [], []
    for i in range(batch_size):
      state = np.array(start)*1
      actions, states = [], []
      for j in range(num_steps):
        if (state[0] < 0 or state[1] < 0 or state[0] > act_ind.shape[1]-1 or
          state[1] > act_ind.shape[2]-1): act = 0
        else: act = act_ind[i, state[0], state[1], state[2]]
        states.append(state*1)
        actions.append(act*1)
        if act == 0: None
        elif act == 1: state[2] = np.mod(state[2]+1, 4)
        elif act == 2: state[2] = np.mod(state[2]-1, 4)
        else:
          if state[2] == 0: state[0] = state[0]+1
          elif state[2] == 1: state[1] = state[1]-1
          elif state[2] == 2: state[0] = state[0]-1
          elif state[2] == 3: state[1] = state[1]+1
      all_actions.append(actions)
      all_states.append(states)
    all_actions = np.array(all_actions)
    all_states = np.array(all_states)
    
    _all_states = all_states
    _all_actions = all_actions
    _state_dist = np.linalg.norm(_all_states[:,:,:2]-np.array(start[:2]), axis=2)*8
    _state_theta = np.arctan2(_all_states[:,:,0]-start[0], _all_states[:,:,1]-start[1])
    _state_orientation = np.mod(-_all_states[:,:,2] + start[2], 4)
    
    # Debug to print distance along traversal.
    # ds = np.ones((batch_size,20), dtype=np.float32)*np.inf
    # for ii in range(batch_size):
    #   for i in range(20): 
    #     _ = all_states[ii,i,:]
    #     if goal_nn[_[0],_[1],_[2]] > 0:
    #       ds[ii,i] = goal_dists[ii][goal_nn[_[0],_[1],_[2]]]
    # print(ds.astype(np.int32))
    
    # states = init_states
    # all_actions, all_states = [], []
    # all_states.append(states)
    # for j in range(batch_size): 
    #   d_starts.append(goal_dists[j][states[j]])
    #   
    # for k in range(num_steps):
    #   acts = []
    #   for j in range(batch_size): 
    #     action_vol = act_ind[j,...]; 
    #     act = action_vol[goal_nn == states[j]]
    #     act = 0 if len(act) == 0 else act[0]
    #     acts.append(act)
    #   states = env.take_action(states, acts)
    #   all_states.append(states)
    #   all_actions.append(acts)
    # all_states = np.array(all_states).T
    # all_actions = np.array(all_actions).T
    # state_dist, state_theta, state_orientation = env.get_relative_coordinates(all_states, init_states)
    return _all_actions, _state_dist, _state_theta, _state_orientation
  
  def execute_actions_1(self, action_volume, init_states, goal_dists, goal_nn, num_steps):
    # Given the action volume around the goal location, walks from some number
    # of start locations by following the maximum action and seeing the final
    # distance to the goal location.
    env = self.task.env
    tp = self.task_params
    tp_ = env.task_params
    batch_size = tp.batch_size
    
    action_volume = np.argmax(action_volume, -1) 
    d_starts, d_ends = [], []
    state_thetas, state_dists = [], []
    all_states = []
    
    states = init_states
    all_states.append(states)
    for j in range(batch_size): 
      d_starts.append(goal_dists[j][states[j]])
      
    for k in range(num_steps):
      acts = []
      for j in range(batch_size): 
        action_vol = action_volume[j,...]; 
        act = action_vol[goal_nn == states[j]]
        act = 0 if len(act) == 0 else act[0]
        acts.append(act);
      states = env.take_action(states, acts)
      all_states.append(states)
    all_states = np.array(all_states).T
    # state_dist, state_theta, state_orientation = \
    #   env.get_relative_coordinates(all_states, self.repeat.ids)
    # state_dists.append(state_dist)
    # state_thetas.append(state_theta)

    d_ends = []
    for j in range(batch_size): 
      d_ends.append(goal_dists[j][states[j]])

    d_ends = np.array(d_ends)
    d_starts = np.array(d_starts)
    # state_dists = np.array(state_dists)*env.task.building.env.resolution
    # state_thetas = np.array(state_thetas)
    return d_starts, d_ends #, (state_dists, state_thetas) 

  def execute_actions(self, action_volume, min_dist=0, num_starts=20):
    # Given the action volume around the goal location, walks from some number
    # of start locations by following the maximum action and seeing the final
    # distance to the goal location.
    env = self.task.env
    graph = self.task.env.task.graph
    nodes = env.task.graph.nodes
    episode = self.episode
    tp = self.task_params
    tp_ = env.task_params
    batch_size = tp.batch_size
    nn_size = self.task.nn_size
 
    num_steps = nn_size*2
    max_dists = nn_size
    rng = np.random.RandomState(0)
    rng_env = np.random.RandomState(0)
    action_volume = np.argmax(action_volume, -1) 
    d_starts, d_ends, all_states, state_thetas, state_dists = [], [], [], [], []
    for i in range(num_starts):
      init_states = [];
      all_states = [];
      for j in range(batch_size):
        goal_dist = episode.all_goal_dists[j]
        starts = np.where(np.logical_and(goal_dist >= min_dist, goal_dist < max_dists))[0]
        if len(starts) == 0:
          starts = np.where(np.logical_and(goal_dist >= 0, goal_dist < max_dists))[0]
        init_state = rng.choice(starts)
        assert(init_state in episode.all_goal_nn[j])
        init_states.append(init_state)
      states = env.reset(rng_env, init_states=init_states, batch_size=None)
      all_states.append(states)
      d_start = []
      for j in range(batch_size): 
        d_start.append(episode.all_goal_dists[j][states[j]])

      for k in range(num_steps):
        acts = []
        for j in range(batch_size): 
          goal_nn = episode.all_goal_nn[j]; action_vol = action_volume[j,...]; 
          act = action_vol[goal_nn == states[j]]
          act = 0 if len(act) == 0 else act[0]
          acts.append(act);
        states = env.take_action(states, acts)
        all_states.append(states)
      all_states = np.array(all_states).T
      state_dist, state_theta, state_orientation = \
        env.get_relative_coordinates(all_states, self.repeat.ids)
      state_dists.append(state_dist)
      state_thetas.append(state_theta)

      # self.repeat.ids
      # Compute distance to goal at end of episode
      d_end = []
      for j in range(batch_size): 
        d_end.append(episode.all_goal_dists[j][states[j]])
      d_ends.append(d_end)
      d_starts.append(d_start)

    d_ends = np.array(d_ends)
    d_starts = np.array(d_starts)
    state_dists = np.array(state_dists)*env.task.building.env.resolution
    state_thetas = np.array(state_thetas)
    return d_starts, d_ends, (state_dists, state_thetas) 

def angle_normalize(x):
  return (((x+np.pi) % (2*np.pi)) - np.pi)

def _test_delta():
  import matplotlib.pyplot as plt
  ref = np.array([[0., 0., 0.]])

  samples = np.random.rand(1000, 3) - 0.5
  samples[:,:2] *= 20
  samples[:,2] *= 2*np.pi

  cost = np.sum(np.square((samples[:,:2] - ref[:,:2]) * 0.1), 1) + angle_normalize(samples[:,2] - ref[:,2])**2
  cost_thresh = 0.25
  
  # Plot the things 
  fig, _, axes = utils.subplot2(plt, (1,1), (10,10))
  ax = axes.pop()
  sz = 10
  for i in range(samples.shape[0]):
    fc = 'b' if cost[i] < cost_thresh else 'r'
    ec = fc
    s = 3
    x, y, t = samples[i,:].tolist()
    ax.arrow(x, y, s*np.cos(t), s*np.sin(t), 
      head_width=0.2, head_length=0.25, fc=fc, ec=ec, alpha=0.5)
  ax.set_xlim([-sz, sz])
  ax.set_ylim([-sz, sz])
  out_file_name = 'tmp/mapper_vis/delta_zone.png'
  fig.savefig(out_file_name, bbox_inches='tight', transparent=True, pad_inches=0)
  plt.close(fig)

def test_mapper_planner():
  from render import swiftshader_renderer as sru
  camera_param = utils.Foo(width=256, height=256, z_near=0.05, z_far=20.0,
    fov_horizontal=60., fov_vertical=60., modalities=['rgb'], img_channels=3,
      im_resize=224./256.)
  r_obj = sru.get_r_obj(camera_param)

  d = factory.get_dataset('sbpd', 'val')
  name = d.get_imset()[0]
  logging.error(name)
  batch_size = 4
  
  top_view_param = tle.get_top_view_discrete_env_task_params(
    prob_random=0.0, batch_size=batch_size, view_scales=[0.125], fovs=[33],
    base_resolution=1.0, step_size=8, top_view=True, ignore_roads=False,
    output_roads=True)
  e = mp_env.MPDiscreteEnv(dataset=d, name=name, task_params=top_view_param,
    flip=False, r_obj=r_obj, rng=np.random.RandomState(0))
  
  mapper_task_params = get_mapper_task_params(
    batch_size=batch_size, num_samples=10, extent_samples=1600,
    output_optimal_actions=True)
  m = MapperPlannerEnv(e, mapper_task_params)
  rng = np.random.RandomState(0)
  import cProfile
  pr = cProfile.Profile()
  pr.enable()
  for i in range(100):
    outputs = m.gen_data(rng)
  pr.disable()
  pr.print_stats(2)
  action_volume = outputs['gt_actions']
  d_starts, d_ends = m.execute_actions(action_volume)
  # np.set_printoptions(2, linewidth=200, threshold=1500)
  # print(outputs['valid'][0,:,:]*1)
  # print((outputs['roads_0'][0,:,:,0]>128)*1)

def _test_mapper():
  from render import swiftshader_renderer as sru
  camera_param = utils.Foo(width=256, height=256, z_near=0.05, z_far=20.0,
    fov_horizontal=60., fov_vertical=60., modalities=['rgb'], img_channels=3,
      im_resize=224./256.)
  r_obj = sru.get_r_obj(camera_param)

  d = factory.get_dataset('sbpd', 'val')
  name = d.get_imset()[0]
  logging.error(name)
  batch_size = 4
  
  top_view_param = tle.get_top_view_discrete_env_task_params(
    prob_random=0.1, batch_size=batch_size, view_scales=[0.5], fovs=[128],
    base_resolution=1.0, step_size=8, top_view=True, ignore_roads=False,
    output_roads=True)
  e = mp_env.MPDiscreteEnv(dataset=d, name=name, task_params=top_view_param,
    flip=False, r_obj=r_obj, rng=np.random.RandomState(0))
  
  mapper_task_params = get_mapper_task_params(
    batch_size=batch_size, num_samples=6, extent_samples=500)
  m = MapperEnv(e, mapper_task_params)
  rng = np.random.RandomState(0)
  outputs = m.gen_data(rng)
  m._vis_data(outputs)

if __name__ == '__main__':
  test_mapper_planner()
