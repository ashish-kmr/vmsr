from __future__ import print_function
import numpy as np, os, cv2, os, scipy, graph_tool as gt, skimage, itertools, logging, copy
from scipy import ndimage
from graph_tool import topology
from src import utils
from src import rotation_utils as ru
from src import map_utils as mu 
from src import graph_utils as gu 
from src import graphs

def get_top_view_env_task_params(noise=0.1, fovs=[128], view_scales=[0.25],
  batch_size=32, ignore_roads=False, output_roads=False, road_dilate_disk_size=0, 
  map_max_size=None, base_resolution=1.0):
  noise_model = utils.Foo(type='gaussian', sigma_trans=noise, sigma_rot=noise, 
    sigma_trans_rot=noise, type_trans='randn', type_rot='randn', 
    type_trans_rot='randn')
  outputs = utils.Foo(top_view=True, top_view_roads=output_roads, loc_on_map=True)
  task_params = utils.Foo(noise_model=noise_model, batch_size=batch_size,
    ignore_roads=ignore_roads, view_scales=view_scales, fovs=fovs,
    outputs=outputs, relight=True, relight_fast=False, 
    road_dilate_disk_size=road_dilate_disk_size, 
    map_max_size=map_max_size, base_resolution=base_resolution)
  return task_params

def take_action_n1(rng, state, action, sigma_rot, sigma_trans, sigma_trans_rot,
  type_rot, type_trans, type_trans_rot):
  """Takes action and generate new state."""
  assert(not(action[0] > 0 and action[1] > 0))
  assert(action[0] >= 0)
  if action[0] > 0:
    # Agent wants to move forward.
    delta = action[0]*np.array([np.cos(state[2]), np.sin(state[2])])
    delta = delta + sigma_trans*action[0]*getattr(rng, type_trans)(2)
    state[:2] = state[:2] + delta
    state[2] = state[2] + sigma_trans_rot*getattr(rng, type_trans_rot)(1)
  if action[1] != 0:
    # Agent wants to rotate in place.
    state[2] = state[2] + action[1] + sigma_rot*action[1]*getattr(rng, type_rot)(1)[0]
  state[2] = np.mod(state[2], 2*np.pi)
  return state
  
def clip_state(final_state, init_state, traversable, step_size):
  # Perform collision checking here, move in the direction given by out_state -
  # state, and clip at a distance before you collide. 
  # Walk along the straight line between init_state and final_state.
  dist = final_state[:2] - init_state[:2]
  r_dist = np.linalg.norm(dist)
  states = np.zeros((0,2), dtype=np.float32)
  if r_dist >= step_size:
    # Check for intermediate locations being valid.
    r_sample = np.arange(0, r_dist, step_size)[:,np.newaxis]
    v_dist = dist / r_dist
    states = init_state[np.newaxis,:2] + r_sample*v_dist[np.newaxis,:]
  
  states = np.concatenate((states, final_state[np.newaxis,:2]), axis=0)
  states = np.round(states).astype(np.int32)
  ind_raise = np.ravel_multi_index((states[:,1], states[:,0]), traversable.shape, mode='wrap')
  ind_clip = np.ravel_multi_index((states[:,1], states[:,0]), traversable.shape, mode='clip')
  ind_inside = ind_raise == ind_clip
  t = traversable.ravel()[ind_clip]
  t = np.logical_and(ind_inside, t)
  ind = np.argmin(t) #Is the index of the first False, otherwise is 0.
  clipped_state = np.concatenate((states[ind-1,:], final_state[2:]), axis=0)
  return clipped_state

def random_walker(rng, iters, num_steps, step_size):
  """Generate random actions."""
  actions = []
  for i in range(iters):
    # Rotate a little bit.
    sign = 1. if rng.rand() > 0.5 else -1.
    action = np.array([0., sign*rng.rand()*np.pi/2])
    actions.append(action)

    # Move forawrd some number of steps.
    num_steps = rng.poisson(lam=4)
    for j in range(num_steps):
      action = np.array([step_size, 0.])
      actions.append(action)
  return actions


class TopViewEnv():
  """Observation is the top-view of the environment.
  Actions are
    - Rotate in place.
    - Move forward x metres.
  """
  def __init__(self, name, dataset, flip, task_params, road=None, view=None, rng=None):
    self.task = utils.Foo()
    if road is not None and view is not None:
      assert(dataset is None)
      assert(flip is None)
      self.task.view = view
      self.task.road = road
    else:
      self.task.view, self.task.road = dataset.load_data(name, flip=flip, 
        map_max_size=task_params.map_max_size, rng=rng, 
        base_resolution=task_params.base_resolution)
    self.task_params = task_params
    self._preprocess_for_task()

  def _preprocess_for_task(self):
    self.take_action_kwargs = {
      'sigma_trans': self.task_params.noise_model.sigma_trans, 
      'sigma_rot': self.task_params.noise_model.sigma_rot,
      'sigma_trans_rot': self.task_params.noise_model.sigma_trans_rot,
      'type_trans': self.task_params.noise_model.type_trans,
      'type_rot': self.task_params.noise_model.type_rot,
      'type_trans_rot': self.task_params.noise_model.type_trans_rot}
    if self.task_params.road_dilate_disk_size > 0:
      disk = skimage.morphology.disk(dtype=np.bool,
        radius=self.task_params.road_dilate_disk_size)
      self.task.road = skimage.morphology.binary_dilation(self.task.road, disk)

    if self.task_params.ignore_roads:
      self.task.traversable = self.task.road == True
      self.task.traversable[:] = True
    else:
      self.task.traversable = self.task.road == True

    self.task.nearest_traversable_dist, self.task.nearest_traversable_ind = \
      scipy.ndimage.morphology.distance_transform_edt(self.task.traversable == False,
      return_distances=True, return_indices=True)
    # Flip to make it x, y.
    self.task.nearest_traversable_ind = self.task.nearest_traversable_ind[::-1,:,:]
    self.task.nearest_traversable_ind = np.transpose(self.task.nearest_traversable_ind, axes=[1,2,0])

    # Resize views with antialiasing.
    self.task.scaled_views = mu.resize_maps(self.task.view,
      self.task_params.view_scales, 'antialiasing')
    self.task.scaled_roads = mu.resize_maps((self.task.road*255).astype(np.uint8),
      self.task_params.view_scales, 'antialiasing')
    self.task.view = None #Takes a lot of memory so remove if not needed.
    self.task.road = None 
  
  def get_loc_axis(self, states):
    """Based on the node orientation returns X, and Y axis. Used to sample the
    map in egocentric coordinate frame.
    """
    loc = states[:,:2]*1.
    theta = states[:,-1:]*1.
    x_axis = np.concatenate((np.cos(theta), np.sin(theta)), axis=1)
    y_axis = np.concatenate((np.cos(theta+np.pi/2.), np.sin(theta+np.pi/2.)),
                            axis=1)
    return loc, x_axis, y_axis, theta

  def reset(self, rng, batch_size=None):
    if batch_size is None:
      batch_size = self.task_params.batch_size
    # Generate seeds for each new episode.
    init_states = []
    episodes = []
    for i in range(batch_size):
      rng_i = np.random.RandomState(rng.randint(np.iinfo(np.uint32).max))
      rng_noise = np.random.RandomState(rng.randint(np.iinfo(np.uint32).max))
      # Initialize the agent somewhere on the map.
      # Randomly sample a point and lookup its nearest traversable ind.
      x = rng_i.randint(self.task.nearest_traversable_ind.shape[1])
      y = rng_i.randint(self.task.nearest_traversable_ind.shape[0])
      x, y = self.task.nearest_traversable_ind[y,x,:]
      assert(self.task.traversable[y,x])
      theta = rng_i.rand()*np.pi*2.
      init_state = np.array([x, y, theta])

      # Reset position
      episode = utils.Foo(rng=rng_i, rng_noise=rng_noise, states=[init_state], 
        executed_actions=[]) 
      episodes.append(episode)
      init_states.append(init_state)
    
    self.episodes = episodes
    # State for the agent is the 2D location on the map, (x,y,theta). 
    return init_states
  
  def take_action(self, states, actions, sim=False):
    """Action is [translation magnitude, rotation magnitude]. 
    """
    out_states = []
    episodes = self.episodes
    batch_size = len(states)
    for i in range(batch_size):
      out_state = take_action_n1(episodes[i].rng_noise, states[i]*1., actions[i],
        **(self.take_action_kwargs))
      out_state = clip_state(out_state, states[i], self.task.traversable, 0.5)

      out_states.append(out_state)
      if not sim:
        episodes[i].states.append(out_state*1.)
        episodes[i].executed_actions.append(actions[i]*1.)
    return out_states
  
  def render_views(self, states):
    # Renders out the view from the state.
    np_states = np.array(states)
    loc, x_axis, y_axis, theta = self.get_loc_axis(np_states)
    views = mu.generate_egocentric_maps(self.task.scaled_views,
      self.task_params.view_scales, self.task_params.fovs, loc, x_axis, y_axis)
    return views

  def get_features(self, states):
    to_output = self.task_params.outputs
    np_states = np.array(states)
    loc, x_axis, y_axis, theta = self.get_loc_axis(np_states)
    
    outputs = {}
    if to_output.top_view:
      views = mu.generate_egocentric_maps(self.task.scaled_views,
        self.task_params.view_scales, 
        self.task_params.fovs, loc, x_axis, y_axis)
      for i, _ in enumerate(views):
        outputs['views_{:d}'.format(i)] = views[i]
    
    if to_output.top_view_roads:
      roads = mu.generate_egocentric_maps(self.task.scaled_roads,
        self.task_params.view_scales, 
        self.task_params.fovs, loc, x_axis, y_axis)
      for i, _ in enumerate(roads):
        outputs['roads_{:d}'.format(i)] = np.expand_dims(roads[i], -1)

    if to_output.loc_on_map:
      loc_on_map = np.array(states)
      for i, sc in enumerate(self.task_params.view_scales):
        outputs['loc_on_map_{:d}'.format(i)] = loc_on_map*1.
        outputs['loc_on_map_{:d}'.format(i)][:,:2] *= sc

    # if to_output.executed_action:
    return outputs

def compute_egomotion(state_from, state_to):
  # Computes the egomotion from state_from to state_to in coordinate frame of
  # the from. dx, dy in coordinate frame of from, d(theta). 
  sf = np.reshape(state_from, [-1,3])
  st = np.reshape(state_to, [-1,3])
  r = np.sqrt(np.sum(np.square(st - sf), axis=1))
  t = np.arctan2(st[:,1] - sf[:,1], st[:,0] - sf[:,0])
  t = t-sf[:,2] + np.pi/2
  dx = np.expand_dims(r*np.cos(t), axis=1)
  dy = np.expand_dims(r*np.sin(t), axis=1)
  dt = np.expand_dims(st[:,2]-sf[:,2], axis=1)
  dt = np.mod(dt, 2*np.pi)
  egomotion = np.concatenate((dx, dy, dt), axis=1)
  egomotion = np.reshape(egomotion, state_from.shape)
  return egomotion

def get_follower_task_params(batch_size=4, gt_delta_to_goal=False,
  terminate_on_complete=False, compute_optimal_actions=False,
  compute_optimal_actions_steps=2, gamma=0.99, add_flips=True, 
  max_dist=200, min_dist=25, act_f=[0., 2., 4., 8., 16.], 
  act_r=[15., 30., 45., 90.], history=0, teacher_steps=20,
  rejection_sampling=False):
  rl_params = utils.Foo(dense_wt=1.0, complete_reward=1.0, time_penalty=-0.01,
    dist_thresh=25., gamma=gamma, terminate_on_complete=terminate_on_complete)
  task_params = utils.Foo(min_dist=min_dist, max_dist=max_dist,
    batch_size=batch_size, act_f=act_f, act_r=act_r, rl_params=rl_params,
    gt_delta_to_goal=gt_delta_to_goal, compute_optimal_actions_step_size=0.9,
    compute_optimal_actions_steps=compute_optimal_actions_steps,
    compute_optimal_actions=compute_optimal_actions, add_flips=add_flips,
    history=history, teacher_steps=teacher_steps,
    rejection_sampling=rejection_sampling)
  return task_params

def _get_discrete_actions(forwards, thetas):
  a_s = []
  for f in forwards: a_s.append([f, 0.])
  for t in thetas: a_s.append([0., np.deg2rad(t)])
  for t in thetas: a_s.append([0., np.deg2rad(-t)])
  a_s = np.array(a_s)
  return a_s

def _precompute_for_optimal_action(actions, num_steps, step_size):
  num_actions = actions.shape[0]
  action_sequences = np.array(list(itertools.product(*([range(num_actions) for _ in range(num_steps)]))))
  
  sz = int(np.round(np.max(actions[:,0]) * num_steps) + 4)
  size = 2*sz + 1
  road = np.zeros((size, size), dtype=np.bool); road[...] = True;
  view = np.zeros((size, size, 3), dtype=np.uint8)
  task_params = get_top_view_env_task_params(noise=0.0)
  e = TopViewEnv(dataset=None, flip=None, name=False, task_params=task_params, 
                     road=road, view=view)

  init_state = e.reset(rng=np.random.RandomState(0), batch_size=action_sequences.shape[0])
  init_state = np.array([(size-1)/2, (size-1)/2, 0.], dtype=np.float32)
  init_states = [init_state for _ in range(action_sequences.shape[0])]

  states = init_states
  all_states = [states]
  for i in range(num_steps):
    states = e.take_action(states, [actions[a] for a in action_sequences[:,i]])
    all_states.append(states)
  all_states = np.array(all_states)

  all_intermediate_states = []
  for i in range(num_steps):
    all_intermediate_states.append(get_intermediate_states(all_states[i+1,:,:], 
      all_states[i,:,:], step_size))

  all_intermediate_states = zip(*all_intermediate_states)
  all_intermediate_states = [np.concatenate(x, axis=0) for x in all_intermediate_states]
  ids = np.concatenate([i*np.ones(_.shape[0]) for i, _ in enumerate(all_intermediate_states)], axis=0)
  ids = ids[:,np.newaxis]
  
  final_ids = [0]
  for i, _ in enumerate(all_intermediate_states): 
    final_ids.append(final_ids[-1]+ _.shape[0])
  final_ids = np.array(final_ids)
  
  ids = np.concatenate([i*np.ones(_.shape[0]) 
      for i, _ in enumerate(all_intermediate_states)], axis=0)[:,np.newaxis]
  all_intermediate_states = np.concatenate(all_intermediate_states, axis=0)
  all_intermediate_states = all_intermediate_states - sz
  final_theta = np.mod(np.sum(actions[action_sequences, 1], 1)[:,np.newaxis], 2*np.pi)
  return all_intermediate_states, ids, final_ids, action_sequences, final_theta

def get_intermediate_states(final_state, init_state, step_size):
  # Perform collision checking here, move in the direction given by out_state -
  # state, and clip at a distance before you collide.
  # Walk along the straight line between init_state and final_state.
  dist = final_state[:,:2] - init_state[:,:2]
  r_dist = np.linalg.norm(dist, axis=1)
  all_intermediate_states = []
  for i in range(final_state.shape[0]):
    states = np.zeros((0,2), dtype=np.float32)
    if r_dist[i] >= step_size:
      # Check for intermediate locations being valid.
      r_sample = np.arange(0, r_dist[i], step_size)[:,np.newaxis]
      v_dist = dist[i,:] / r_dist[i]
      states = init_state[i,:][np.newaxis,:2] + r_sample*v_dist[np.newaxis,:]
    states = np.concatenate((states, final_state[i,:][np.newaxis,:2]), axis=0)
    all_intermediate_states.append(states)
  return all_intermediate_states

class Follower():
  """Provides data for the follower leader style problem. The leader generates
  targets, and outputs the set of images seen in going from the starting
  location to the goal location. Follower has noisy actuators and has to be
  able to get to the location of the target picked out by the leader."""
  
  def __init__(self, env, task_params):
    self.task = utils.Foo()
    self.task.env = env
    self.task.graph = graphs.OrientedGridGraph(env.task.traversable)
    
    self.task_params = task_params
    self.task.actions = _get_discrete_actions(self.task_params.act_f,
      self.task_params.act_r)
    if self.task_params.rejection_sampling:
      self.task.graph.setup_rejection_sampling(self.task_params.min_dist,
        self.task_params.max_dist, 20, 200, 100,
        target_d='uniform')
    
    if self.task_params.compute_optimal_actions:
      optimal = utils.Foo()
      optimal.canonical_states, optimal.state_ids, optimal.final_ids, optimal.action_sequences, optimal.final_theta = \
        _precompute_for_optimal_action(self.task.actions, 
          self.task_params.compute_optimal_actions_steps, 
          self.task_params.compute_optimal_actions_step_size)
      self.task.optimal = optimal
 
  def reset(self, rng):
    start_nodes = []; end_nodes = []; d_inits = []; node_dists = [];
    # Generates the problem.
    for i in range(self.task_params.batch_size):
      if self.task_params.rejection_sampling:
        start_node, end_node, node_dist, d_init = self.task.graph.sample_random_path_rejection(rng, 
          min_dist=self.task_params.min_dist, max_dist=self.task_params.max_dist)
      else:
        start_node, end_node, node_dist, d_init = self.task.graph.sample_random_path(rng, 
          min_dist=self.task_params.min_dist, max_dist=self.task_params.max_dist)
      start_nodes.append(start_node)
      end_nodes.append(end_node)
      node_dists.append(node_dist)
      d_inits.append(d_init)
    
    self.task.node_dists = node_dists 
    self.task.d_inits = d_inits

    # Set up problem for the follower agent. 
    _ = self.task.env.reset(rng, batch_size=self.task_params.batch_size)
    init_states = []; goal_states = []; completed = []; rngs = [];
    for i in range(self.task_params.batch_size):
      init_node = np.zeros(3, dtype=np.float32)
      init_node[:2] = self.task.graph.nodes[start_nodes[i], :2]
      init_node[2] = (rng.rand()-0.5)*np.pi/2. + np.pi/2. * self.task.graph.nodes[start_nodes[i], 2]
      self.task.env.episodes[i].init_state = init_node
      
      goal_node = np.zeros(3, dtype=np.float32)
      goal_node[:2] = self.task.graph.nodes[end_nodes[i], :2]
      goal_node[2] = (rng.rand()-0.5)*np.pi/2. + np.pi/2. * self.task.graph.nodes[end_nodes[i], 2]
      
      rngs.append(np.random.RandomState(rng.randint(np.iinfo(np.uint32).max)))
      goal_states.append(goal_node)
      init_states.append(init_node)
      completed.append(False)
    
    self.task.history_f = [] 
    self.task.goal_states = goal_states 
    self.task.completed = completed
    self.task.init_states = init_states
    self.task.rngs = rngs
    return init_states
  
  def _get_teacher_trajectory(self, rngs):
    # Compute the ground truth trajectory on the tasks (called by
    # get_common_data to generate data for agent 1).
    all_states = []
    all_actions = []
    states = self.task.init_states
    all_states.append(states)
    for i in range(self.task_params.teacher_steps):
      action_probs = self.get_optimal_action(states, i)
      sampled_action = [np.argmax(rngs[j].multinomial(1, action_prob)) 
        for j, action_prob in enumerate(action_probs)]
      all_actions.append(sampled_action)
      sampled_action_ = [self.task.actions[_]*1. for _ in sampled_action]
      states = self.task.env.take_action(states, sampled_action_, sim=True)
      all_states.append(states)
    all_actions = np.array(all_actions).T
    if all_actions.size == 0:
      all_actions = np.zeros((len(rngs), 0), dtype=np.int32)
    all_states = np.transpose(np.array(all_states), axes=[1,0,2])
    return all_actions, all_states

  def get_common_data(self):
    # Render out the images that the mapper can use to move around, lets sample
    # the mid point image.
    
    # Generate the target view of the world.
    outputs = {}
    goal_views = self.task.env.render_views(self.task.goal_states)
    outputs['target_view'] = np.expand_dims(goal_views[0], axis=1)

    teacher_actions, teacher_states = self._get_teacher_trajectory(self.task.rngs)
    outputs['teacher_actions'] = teacher_actions
    teacher_states = teacher_states[:,:-1,:]

    # Render out the views.
    intermediate_views = self.task.env.render_views(teacher_states.reshape(-1,3))[0]
    sh = [teacher_states.shape[0], teacher_states.shape[1],
      intermediate_views.shape[1], intermediate_views.shape[2],
      intermediate_views.shape[3]]
    intermediate_views = np.reshape(intermediate_views, sh)
    outputs['teacher_views'] = intermediate_views.astype(np.int32)

    outputs['full_view'] = self.task.env.task.scaled_views[0]
    outputs['full_road'] = np.expand_dims(self.task.env.task.scaled_roads[0], -1)
    
    i = 0; sc = self.task.env.task_params.view_scales[i];
    goal_loc = np.array(self.task.goal_states)
    goal_loc[:,:2] *= sc
    outputs['goal_loc'] = np.expand_dims(goal_loc, axis=1) 
    
    init_states = [e.init_state for e in self.task.env.episodes]
    e = compute_egomotion(np.array(init_states), np.array(self.task.goal_states))
    theta = np.concatenate((np.cos(e[:,-1:]), np.sin(e[:,-1:])), axis=1)
    f = np.concatenate((e[:,:2], theta), axis=1)
    outputs['init_delta_to_goal'] = np.expand_dims(f, axis=1)
    return outputs
  
  def pre_common_data(self, inputs):
    return inputs

  def pre_features(self, f):
    return f
  
  def get_distance_to_goal(self, states):
    dists = []
    h_dists = np.linalg.norm(np.abs(np.array(self.task.goal_states)-np.array(states))[:,:2], axis=1)
    h_dists = h_dists.tolist()
    for i, state in enumerate(states):
      # Rotate and place at the node location.
      quant_theta = np.mod(np.round(state[2]).astype(np.int32), 4)
      node_id = np.round(state[:2]).astype(np.int32)
      node_id = self.task.graph.node_ids[node_id[1], node_id[0]]
      node_id_theta = node_id + quant_theta * (self.task.graph.nodes.shape[0]/4)
      state_distance = self.task.node_dists[i][node_id_theta]*1
      dists.append(state_distance)
    return dists, h_dists 
  
  def get_features(self, states, step_number=None):
    f = self.task.env.get_features(states)
    while len(self.task.history_f) < self.task_params.history:
      self.task.history_f.insert(0, copy.deepcopy(f))
    # Insert the latest frame.
    self.task.history_f.insert(0, copy.deepcopy(f))
    
    view = np.concatenate([np.expand_dims(x['views_0'], -1) for x in self.task.history_f], -1)
    view = np.expand_dims(view, axis=1)
    f['view'] = view
    if 'roads_0' in f:
      road = np.concatenate([np.expand_dims(x['roads_0'], -1) for x in self.task.history_f], -1)
      road = np.expand_dims(road, axis=1)
      f['road'] = road
    
    f['loc_on_map'] = np.expand_dims(f['loc_on_map_0'], axis=1)
    dists, h_dists = self.get_distance_to_goal(states)
    f['dist_to_goal'] = np.expand_dims(np.array(dists), axis=1).astype(np.float32)
    f['h_dist_to_goal'] = np.expand_dims(np.array(h_dists), axis=1).astype(np.float32)
    if self.task_params.gt_delta_to_goal:
      e = compute_egomotion(np.array(states), np.array(self.task.goal_states))
      theta = np.concatenate((np.cos(e[:,-1:]), np.sin(e[:,-1:])), axis=1)
      f['gt_delta_to_goal'] = np.concatenate((e[:,:2], theta), axis=1)
      f['gt_delta_to_goal'] = np.expand_dims(f['gt_delta_to_goal'], axis=1)
    f.pop('views_0', None); f.pop('roads_0', None); f.pop('loc_on_map_0', None);
    self.task.history_f.pop()
    return f

  def take_action(self, states, actions, step_number=None):
    rl_params = self.task_params.rl_params
    _actions = [self.task.actions[i]*1. for i in actions]
    new_states = self.task.env.take_action(states, _actions)
    rewards = []
    for i, (s0, s1) in enumerate(zip(states, new_states)):
      s0 = np.round(s0).astype(np.int32) 
      n0 = self.task.graph.node_ids[s0[1], s0[0]]
      s1 = np.round(s1).astype(np.int32) 
      n1 = self.task.graph.node_ids[s1[1], s1[0]]
      d0 = self.task.node_dists[i][n0]
      d1 = self.task.node_dists[i][n1]
      r = 0. 
      if not (self.task.completed[i] and self.task_params.rl_params.terminate_on_complete):
        if d1 < rl_params.dist_thresh: 
          r += rl_params.complete_reward
          self.task.completed[i] = True
        r = r + rl_params.dense_wt * (d0*1. - d1*self.task_params.rl_params.gamma) / self.task.d_inits[i]
        r = r + rl_params.time_penalty
      rewards.append(r)
    return new_states, rewards

  def get_optimal_action(self, states, j):
    a = np.zeros((len(states), self.task.actions.shape[0]), dtype=np.float32)
    MAX_DIST = 123456
    seq_distance = None
    if self.task_params.compute_optimal_actions:
      for i, state in enumerate(states):
        # Rotate and place at the node location.
        final_theta = self.task.optimal.final_theta + state[2]
        quant_theta = np.mod(np.round(final_theta/(np.pi/2.)).astype(np.int32), 4)
        R = ru.get_r_matrix_2d(state[2])
        intermediate_states = np.round(np.dot(self.task.optimal.canonical_states, R.T) + state[np.newaxis,:2]).astype(np.int32)
        # Lookup the node_id for this state, if node_id is invalid for any
        # state then that is invalid, otherwise lookup the last state. 
        node_ids = get_array_value_at_subs(self.task.graph.node_ids, 
          (intermediate_states[:,1], intermediate_states[:,0]), -1)
        invalid = node_ids < 0
        seq_invalid = np.add.reduceat(invalid, self.task.optimal.final_ids[:-1])
        
        final_node_id = node_ids[self.task.optimal.final_ids[1:]-1]*1
        final_node_id[final_node_id < 0] = 0

        final_node_id_theta = final_node_id + quant_theta[:,0] * (self.task.graph.nodes.shape[0]/4)
        seq_distance = self.task.node_dists[i][final_node_id]*1

        # final_node_id = node_ids[self.task.optimal.final_ids[1:]-1]*1
        # neigh_node_id = get_array_value_at_subs(self.task.graph.neighbor_ids[final_node_id,:], (np.arange(quant_theta.shape[0]), quant_theta[:,0]), -1)
        # n_distance = self.task.node_dists[i][neigh_node_id]*1
        # n_distance[neigh_node_id < 0] = MAX_DIST
        # seq_distance = np.minimum(seq_distance+2, n_distance+1)

        seq_distance[seq_invalid > 0] = MAX_DIST
        # sq = seq_distance*1.
        # sq = -sq; sq = sq - np.max(sq)
        # sq = np.exp(sq); sq = sq / (np.sum(sq)+1e-3)
        # a[i, self.task.optimal.action_sequences[:,0]] = sq
        
        # ind = seq_distance == np.min(seq_distance)
        # optimal_action = np.unique(self.task.optimal.action_sequences[ind,0])
        # a[i,optimal_action] = 1

        sq_ = -seq_distance*1.;
        sq_ = np.reshape(sq_, [self.task.actions.shape[0], -1])
        sq_ = np.max(sq_, axis=1)
        sq_ = sq_- np.max(sq_); 
        sq_ = np.exp(sq_/1.); 
        sq_ = sq_ / (np.sum(sq_)+1e-3);
        a[i,:] = sq_
    else:
      a[:,0] = 1
    self.task.optimal_action = a*1
    return a 

  def get_targets(self, states, j):
    # a = np.zeros((len(states), 1, self.task.actions.shape[0]), dtype=np.int32);
    # a[:,:,0] = 1;
    a = np.expand_dims(self.task.optimal_action, axis=1)*1
    return {'gt_action': a}

  def get_gamma(self):
    return self.task_params.rl_params.gamma

class EnvMultiplexer():
  # Samples an environment at each iteration.
  def __init__(self, args):
    params = vars(args)
    for k in params.keys():
      setattr(self, k, params[k])
    self._pick_data()
    self._setup_data()

  def _pick_data(self):
    # Re does self.names to only train on data this worker should train on.
    names = [(x, False) for x in self.names]
    if self.follower_task_params.add_flips:
      names += [(x, True) for x in self.names]
    while len(names) < self.num_workers:
      logging.error('#Env: %d, #workers: %d', len(names), self.num_workers)
      names = names + names
    
    to_pick = range(self.worker_id, len(names), self.num_workers)
    logging.error('All Data: %s', str(names))
    logging.error('worker_id: %d, num_workers: %d', self.worker_id, self.num_workers)
    logging.error('Picking data: %s', str(to_pick))

    self.names = [names[i] for i in to_pick]
    logging.error('Picked Data: %s', str(self.names))
  
  def _setup_data(self):
    # Load building env class.
    es = []
    logging.error('_setup_data')
    for b, flip in self.names:
      logging.error('Loading %s with flip %d.', b, flip)
      e = TopViewEnv(b, self.dataset, flip, self.top_view_task_params, 
        rng=np.random.RandomState(0))
      obj = Follower(e, self.follower_task_params)
      es.append(obj)
      logging.error('Loaded %s with flip %d.', b, flip)
    self.envs = es

  def sample_env(self, rng):
    env_id = rng.choice(len(self.envs))
    return self.envs[env_id]

def get_road_prediction_task_params(batch_size=4, add_flips=True):
  task_params = utils.Foo(batch_size=batch_size, add_flips=add_flips)
  return task_params

class RoadMultiplexer():
  def __init__(self, args):
    params = vars(args)
    for k in params.keys():
      setattr(self, k, params[k])
    self._setup_data()
  
  def _setup_data(self):
    es = []
    flips = [False]
    if self.road_prediction_task_params.add_flips: flips = [True, False]

    for b in self.names:
      for flip in flips:
        e = TopViewEnv(b, self.dataset, flip, self.top_view_task_params)
        y, x = np.where(e.task.traversable)
        nodes = np.array([x,y]).T
        e.task.nodes = nodes
        es.append(e)
    self.envs = es
  
  def gen_data(self, rng):
    per_env = self.road_prediction_task_params.batch_size / len(self.envs)
    view, road = [], []
    for i, e in enumerate(self.envs):
      # Sample points on the road randomly, along with a random rotation.
      ids = rng.choice(e.task.nodes.shape[0], per_env)
      nodes = e.task.nodes[ids,:]*1.
      nodes = nodes + rng.rand(per_env, 2) - 0.5
      theta = rng.rand(per_env, 1) * np.pi * 2.
      states = np.concatenate((nodes, theta), axis=1)
      states = np.split(states, states.shape[0], axis=0)
      states = [s[0] for s in states]
      f = e.get_features(states)
      view.append(f['views_0'])
      road.append(f['roads_0'])
    view = np.concatenate(view, axis=0)
    road = (np.concatenate(road, axis=0) > 128).astype(np.int32)
    outputs = {'view': view, 'label': road}
    return outputs

class OdoMultiplexer():
  def __init__(self, args):
    params = vars(args)
    for k in params.keys():
      setattr(self, k, params[k])
    self._setup_data()
  
  def _setup_data(self):
    # Load building env class.
    es = []
    for b in self.names:
      e = TopViewEnv(b, self.dataset, False, self.task_params)
      obj = Odometry(e)
      es.append(obj)
      e = TopViewEnv(b, self.dataset, True, self.task_params)
      obj = Odometry(e)
      es.append(obj)
    self.envs = es

  def gen_data(self, rng):
    data = []
    for e in self.envs:
      data.append(e.gen_data(rng))
    data_all = data[0]
    for d in data[1:]:
      data_all = self.envs[0].concat_data(data_all, d)
    return data_all

class Odometry():
  def __init__(self, env):
    self.env = env
    self.num_steps = 6
    self.set_rng_seed(0)
  
  def set_rng_seed(self, seed):
    self.rng = np.random.RandomState(seed)

  def gen_data(self, rng=None):
    if rng is None:
      rng = self.rng
    env = self.env
    
    num_steps = self.num_steps 
    n_scales = len(env.task_params.fovs)

    # Generate data for visual odometry.
    outputs = {}
    init_states = env.reset(rng)
    actions_ = []
    for init_state in init_states:
      action = random_walker(rng, iters=10, num_steps=5, step_size=40)
      actions_.append(action)

    actions = []
    for j in range(num_steps):
      action = []
      for i in range(len(init_states)):
        action.append(actions_[i][j])
      actions.append(action)
    
    states = init_states; views = [];
    states_all = [];
    
    for k in range(n_scales):
      views.append([])
    
    states_all = []
    
    fs = env.get_features(states)
    states_all.append(states)
    for k in range(n_scales):
      views[k].append(fs['views_{:d}'.format(k)]) 
    
    for i in range(num_steps):
      states = env.take_action(states, actions[i])
      fs = env.get_features(states)
      states_all.append(states)
      for k in range(n_scales):
        views[k].append(fs['views_{:d}'.format(k)]) 
    
    states_all = np.array(states_all) # (T+1) x B x 3
    egomotion = compute_egomotion(states_all[:-1,...], states_all[1:,...])

    befores = [None for _ in range(n_scales)]
    afters = [None for _ in range(n_scales)]
    for k in range(n_scales):
      views[k] = np.array(views[k]) # (T+1) x B x H x W x 3
      # Generate view pairs.
      after = views[k][1:,...]     # T x B x H x W x 3 
      before = views[k][:-1,...]   # T x B x H x W x 3
        
      # Reshape
      sh = before.shape
      before = np.reshape(before, [-1] + list(sh)[2:])
      after = np.reshape(after, [-1] + list(sh)[2:])

      outputs['view_before_{:d}'.format(k)] = before
      outputs['view_after_{:d}'.format(k)] = after
    # Egomotion will be the motion of the agent. 
    outputs['egomotion'] = np.reshape(egomotion, [-1,3])
    outputs['state_before'] = np.reshape(states_all[:-1,...]*1., [-1,3])
    outputs['state_after'] = np.reshape(states_all[1:,...]*1., [-1,3])
    return outputs

  def concat_data(self, data1, data2):
    # Concats data from data1 and data2.
    ks = data1.keys()
    data = {}
    for k in ks:
      data[k] = np.concatenate((data1[k], data2[k]), axis=0)
    return data

def get_array_value_at_subs(a, subs, invalid_subs_value):
  ind_clip = np.ravel_multi_index(subs, a.shape, mode='clip')
  ind_wrap = np.ravel_multi_index(subs, a.shape, mode='wrap')
  vals = a.ravel()[ind_clip]
  vals[ind_clip != ind_wrap] = invalid_subs_value
  return vals

# class GridGraph():
#   def __init__(self, traversable):
#     """Given a traversable map, computes an undirected grid graph on the
#     space."""
#     # Canonicalize the nodes.
#     assert(np.sum(traversable) < 0.25*traversable.size), \
#       'Doesn"t look like a sparse graph {:d} / {:d}'.format(np.sum(traversable), traversable.size)
#     y, x = np.where(traversable)
#     nodes = np.array([x,y]).T
#     ind = np.argsort(nodes[:,1], kind='mergesort')
#     nodes = nodes[ind, :]
#     ind = np.argsort(nodes[:,0], kind='mergesort')
#     nodes = nodes[ind,:]
# 
#     # +y-edges
#     conn = np.all(nodes[1:,:] - nodes[:-1,:] == np.array([[0,1]]), axis=1)
#     first = np.where(conn)[0]
#     edges_y = np.array([first, first+1]).T
# 
#     # +x-edges
#     ind1 = np.argsort(nodes[:,0], kind='mergesort')
#     n = nodes[ind1,:]
#     ind2 = np.argsort(nodes[:,1], kind='mergesort')
#     n = nodes[ind2,:]
#     ind = ind1[ind2]
#     assert(np.all(n == nodes[ind,:]))
# 
#     conn = np.all(n[1:,:] - n[:-1,:] == np.array([[1,0]]), axis=1)
#     first = np.where(conn)[0]
#     edges_x = np.array([ind[first], ind[first+1]]).T
#     
#     g = gt.Graph(directed=False)
#     g.add_vertex(n=nodes.shape[0])
#     g.add_edge_list(edges_x)
#     g.add_edge_list(edges_y)
#     
#     # Label and prune away empty clusters in graph.
#     comps = gt.topology.label_components(g)
#     
#     # Code to lookup the node it from the image
#     node_ids = -1*np.ones(traversable.shape, dtype=np.int32)
#     aa = np.ravel_multi_index((nodes[:,1], nodes[:,0]), node_ids.shape)
#     node_ids.ravel()[aa] = np.arange(aa.shape[0])
#     
#     self.graph = g
#     self.nodes = nodes
#     self.node_ids = node_ids
#     self.component = np.array(comps[0].get_array())
#     self.component_counts = comps[1]*1
#     
#     self.compute_neighbor_indices()
# 
#   def compute_neighbor_indices(self):
#     neighbor_ids = -1*np.ones((self.nodes.shape[0], 4), dtype=np.int32)
#     # Corresponding to orientation being 0, 90, 180, 270
#     neighbor_ids_dense = -1*np.ones_like(self.node_ids)
#     
#     neighbor_ids_dense[:] = -1; ids = 0;
#     neighbor_ids_dense[:,:-1] = self.node_ids[:,1:]
#     ind = self.node_ids.ravel() >= 0
#     neighbor_ids[self.node_ids.ravel()[ind], ids] = neighbor_ids_dense.ravel()[ind]
#     
#     neighbor_ids_dense[:] = -1; ids = 2;
#     neighbor_ids_dense[:,1:] = self.node_ids[:,:-1]
#     ind = self.node_ids.ravel() >= 0
#     neighbor_ids[self.node_ids.ravel()[ind], ids] = neighbor_ids_dense.ravel()[ind]
#   
#     neighbor_ids_dense[:] = -1; ids = 1;
#     neighbor_ids_dense[:-1,:] = self.node_ids[1:,:]
#     ind = self.node_ids.ravel() >= 0
#     neighbor_ids[self.node_ids.ravel()[ind], ids] = neighbor_ids_dense.ravel()[ind]
#     
#     neighbor_ids_dense[:] = -1; ids = 3;
#     neighbor_ids_dense[-1:,:] = self.node_ids[:1,:]
#     ind = self.node_ids.ravel() >= 0
#     neighbor_ids[self.node_ids.ravel()[ind], ids] = neighbor_ids_dense.ravel()[ind]
# 
#     self.neighbor_ids = neighbor_ids
#   
#   def sample_random_path(self, rng, min_dist=100, max_dist=3200):
#     # Sample a component based on size.
#     id1 = rng.choice(self.nodes.shape[0], size=1)
#     dist, pred_map = gt.topology.shortest_distance(gt.GraphView(self.graph, reversed=True), 
#       source=id1, target=None, max_dist=max_dist*2, pred_map=True)
#     node_dist = np.array(dist.get_array())
#     node_dist[node_dist >= max_dist*2] = 2*max_dist
#     
#     ids = np.where(np.logical_and(node_dist >= min_dist, node_dist <= max_dist))[0]
#     if ids.size > 0:
#       id2 = rng.choice(ids)
#       path = gt.topology.shortest_path(gt.GraphView(self.graph, reversed=True), 
#         source=id1, target=self.graph.vertex(id2), pred_map=pred_map)
#       path_node_ids = [int(x) for x in path[0]]
#       path_node_ids = np.array(path_node_ids)[::-1]
#       return path_node_ids, node_dist, node_dist[id2]
#     else:
#       return self.sample_random_path(rng, min_dist, max_dist)
