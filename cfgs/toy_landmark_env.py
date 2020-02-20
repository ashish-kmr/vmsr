from __future__ import print_function
import logging
import numpy as np, os, cv2, os, scipy, graph_tool as gt, skimage, itertools, copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from graph_tool import topology
from src import utils
from src import rotation_utils as ru
from src import map_utils as mu 
from src import graph_utils as gu 
from src import graphs as graphs 
from copy import deepcopy
#from env.mp_env import expert_navigator

def get_top_view_discrete_env_task_params(prob_random=0.0, fovs=[128], view_scales=[0.25],
  batch_size=32, ignore_roads=True, output_roads=False, road_dilate_disk_size=0,
  map_max_size=None, base_resolution=1.0, step_size=1, top_view=False, perturb_views=False,
  graph_type='ogrid',t_prob_noise=0.0,task_typ='forward',replay_buffer=0,tf_distort=False,
  minQ=20,reuseCount=4,spacious=False,multi_act=False, dilation_cutoff=4):
  noise_model = utils.Foo(prob_random=prob_random)
  outputs = utils.Foo(top_view=top_view, top_view_roads=output_roads, loc_on_map=True)
  task_params = utils.Foo(noise_model=noise_model, batch_size=batch_size,
    ignore_roads=ignore_roads, view_scales=view_scales, fovs=fovs,
    outputs=outputs, relight=True, relight_fast=False,
    road_dilate_disk_size=road_dilate_disk_size,
    map_max_size=map_max_size, base_resolution=base_resolution, step_size=step_size,
    perturb_views=perturb_views, graph_type=graph_type,t_prob_noise=t_prob_noise,
    replay_buffer=replay_buffer,tf_distort=tf_distort,minQ=minQ,reuseCount=reuseCount,
    spacious=spacious,multi_act=multi_act, dilation_cutoff=dilation_cutoff)
  assert(graph_type in ['ogrid', 'null'])
  return task_params

def _get_relative_goal_loc(goal_loc, loc, theta):
  r = np.sqrt(np.sum(np.square(goal_loc - loc), axis=1))
  t = np.arctan2(goal_loc[:,1] - loc[:,1], goal_loc[:,0] - loc[:,0])
  t = t-theta[:,0] + np.pi/2
  return np.expand_dims(r,axis=1), np.expand_dims(t, axis=1)

def perturb_images(imgs, rngs, noise):
  # Given a list of images and a list of rngs, perturb the images.
  ff = 0.5; imgs_out = [];
  for i in range(len(rngs)):
    rng = rngs[i]
    img = imgs[i,...]*1
    
    # Perturb all images in this set.
    '''if rng.rand() < ff:
      # Zoom in and rotate.
      image_center = tuple(np.array(img.shape[:2])/2)
      angle = 2*(rng.rand()-0.5)*90*noise
      scale = np.exp(rng.rand()*np.log(1.+noise))
      rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
      img = cv2.warpAffine(img, rot_mat, img.shape[:2], flags=cv2.INTER_LINEAR)
    '''
    
    if rng.rand() < ff and img.shape[2] == 3:
      # Only messes with color images
      # Mess with the channels a little bit.
      w = np.exp(2*(rng.rand(1,1,3)-0.5)*np.log(1.+noise))
      img = img*w
      img = img.astype(np.uint8)
    
    imgs_out.append(img)
  return np.array(imgs_out)

class EnvMultiplexer():
  # Samples an environment at each iteration.
  def __init__(self, args, worker_id=0, num_workers=1):
    params = vars(args)
    self.r_obj = None
    for k in params.keys():
      setattr(self, k, params[k])
    self._pick_data(worker_id, num_workers)
    self._setup_data()
    self.batch = -1

  def _pick_data(self, worker_id, num_workers):
    # Re does self.names to only train on data this worker should train on.
    names = [(x, False) for x in self.names]
    if self.env_task_params_2.add_flips:
      names += [(x, True) for x in self.names]
    out_names = []
    if len(names) < num_workers:
      while len(out_names) < num_workers:
        logging.error('#Env: %d, #workers: %d', len(names), num_workers)
        out_names = out_names + names
      names = out_names[:num_workers]
    
    to_pick = range(worker_id, len(names), num_workers)
    logging.error('All Data: %s', str(names))
    logging.error('worker_id: %d, num_workers: %d', worker_id, num_workers)
    logging.error('Picking data: %s', str(to_pick))

    self.names = [names[i] for i in to_pick]
    logging.error('Picked Data: %s', str(self.names))
  
  def _setup_data(self):
    # Load building env class.
    es = []
    # Setup renderer if necessary.
    if self.camera_param: r_obj = self.get_r_obj(self.camera_param)
    else: r_obj = None
    
    for b, flip in self.names:
      logging.error('Loading %s with flip %d.', b, flip)
      e = self.env_class(b, self.dataset, flip, self.env_task_params,
        rng=np.random.RandomState(0), r_obj=r_obj)
      obj = self.env_class_2(e, self.env_task_params_2)
      es.append(obj)
      logging.error('Loaded %s with flip %d.', b, flip)
    self.envs = es
    # Kill the renderer
    self.r_obj = None

  def sample_env(self, rng):
    env_id = rng.choice(len(self.envs))
    self.batch = self.batch+1
    self.envs[env_id].batch = self.batch
    return self.envs[env_id], env_id

  def get_env(self, env_id):
    return self.envs[env_id]
  
  def gen_data(self, rng):
    """Used for generating data for a simple CNN."""
    env_id = rng.choice(len(self.envs))
    e = self.envs[env_id]
    self._last_env = self.envs[env_id]
    self.batch = self.batch+1
    e.batch = self.batch
    return e.gen_data(rng)
    
  def get_r_obj(self, camera_param):
    if self.r_obj is None:
      from render import swiftshader_renderer as sru
      cp = camera_param
      rgb_shader, d_shader = sru.get_shaders(cp.modalities)
      r_obj = sru.SwiftshaderRenderer()
      fov_vertical = cp.fov_vertical
      r_obj.init_display(width=cp.width, height=cp.height,
        fov_vertical=fov_vertical, fov_horizontal=cp.fov_horizontal,
        z_near=cp.z_near, z_far=cp.z_far, rgb_shader=rgb_shader,
        d_shader=d_shader, im_resize=cp.im_resize)
      r_obj.clear_scene()
      self.r_obj = r_obj
    return self.r_obj

class DiscreteEnv():
  """Observation is the top-view of the environment.
  Actions are simple grid world actions.
    - Rotate left, right, move straight stay in place.
    - With some probability it stays in place.
  """
  def __init__(self, task_params):
    # The Expectation is the fill this function with code to fill up task and
    # task_params.
    raise NotImplemented

  def _setup_noise(self):
    self.take_action_kwargs = {
      'prob_random': self.task_params.noise_model.prob_random } 
  
  def _compute_graph(self):
    """Computes traversibility and then uses it to compute the graph."""
    if self.task_params.road_dilate_disk_size > 0:
      disk = skimage.morphology.disk(dtype=np.bool,
        radius=self.task_params.road_dilate_disk_size)
      self.task.road = skimage.morphology.binary_dilation(self.task.road, disk)
    
    if self.task_params.ignore_roads:
      self.task.traversable = self.task.road == True
      self.task.traversable[:] = True
    else:
      self.task.traversable = self.task.road == True
    #print(self.task.traversable.shape)
    # Build a grid graph on space for fast shortest path queries.
    if self.task_params.graph_type == 'null':
      self.task.graph = graphs.NullGraph(self.task.traversable, True, 
        self.task_params.step_size, 0) 
    elif self.task_params.graph_type == 'ogrid':
      self.task.graph = graphs.OrientedGridGraph(self.task.traversable, 
        force=True, step_size=self.task_params.step_size)

    # Compute embedding for nodes, compute an id for x coordinate, y-coordinate and theta.
    node_embedding = np.zeros((self.task.graph.nodes.shape[0], 3), dtype=np.int32)
    for i in range(3): 
      _, node_embedding[:,i] = np.unique(self.task.graph.nodes[:,i], return_inverse=True)
    self.task.node_embedding = node_embedding

  def get_loc_axis(self, states):
    """Based on the node orientation returns X, and Y axis. Used to sample the
    map in egocentric coordinate frame.
    """
    loc = states[:,0:2]*1.
    theta = states[:,-1:]*np.pi/2.
    x_axis = np.concatenate((np.cos(theta), np.sin(theta)), axis=1)
    y_axis = np.concatenate((np.cos(theta+np.pi/2.), np.sin(theta+np.pi/2.)),
                            axis=1)
    return loc, x_axis, y_axis, theta

  def get_relative_coordinates(self, target_states, ref_states):
    """Given reference nodes (and not ids) [N] returns the relative
    coordinates of the targets [N x K] wrt reference nodes."""
    loc, x_axis, y_axis, theta = self.get_loc_axis(ref_states)
    #print(target_states.shape)
    rel_goal_orientation_, goal_dist_, goal_theta_ = [], [], []
    for i in range(target_states.shape[1]):
      goal_loc, _, _, _theta = self.get_loc_axis(np.array(target_states[:,i]))
      rel_goal_orientation = 4*np.mod(theta-_theta, 2*np.pi) / (2*np.pi)
      # rel_goal_orientation = np.mod(np.int32((theta - _theta)/(np.pi/2)), 4)
      goal_dist, goal_theta = _get_relative_goal_loc(goal_loc, loc, theta)
      
      goal_theta_.append(goal_theta)
      rel_goal_orientation_.append(rel_goal_orientation)
      goal_dist_.append(goal_dist)
    
    goal_dist = np.array(goal_dist_)[...,0].T
    goal_theta = np.array(goal_theta_)[...,0].T
    rel_goal_orientation = np.array(rel_goal_orientation_)[...,0].T
    #print(goal_dist.shape, goal_theta.shape, rel_goal_orientation.shape)
    return goal_dist, goal_theta, rel_goal_orientation

  def reset(self, rng, init_states=None, batch_size=None):
    if batch_size is None:
      batch_size = self.task_params.batch_size
    assert(init_states is None or batch_size == len(init_states))

    episodes = []
    out_init_states = []
    for i in range(batch_size):
      # Generate seeds for each new episode.
      rng_i = np.random.RandomState(rng.randint(np.iinfo(np.uint32).max))
      rng_noise = np.random.RandomState(rng.randint(np.iinfo(np.uint32).max))
      
      # Initialize the agent somewhere on the map (grid location and a fixed
      # orientation).
      if init_states is None:
        waypoints, path = self.task.graph.sample_random_path_waypoints(rng_i, 1,
                min_dist=4, max_dist=200)
        init_state = waypoints[0]
      else:
        init_state = init_states[i]

      # Reset position
      episode = utils.Foo(rng=rng_i, rng_noise=rng_noise, states=[init_state],
        executed_actions=[], action_status=[]) 
      episodes.append(episode)
      out_init_states.append(init_state)
    
    self.episodes = episodes
    # State for the agent is the 2D location on the map, (x,y,theta). 
    return out_init_states
  
  def take_action(self, states, actions, sim=False):
    """Actions are discrete [0 (stay in place), 1(turn left), 2(turn right),
    3(straight ahead)].
    """
    out_states = []
    episodes = self.episodes
    batch_size = len(states)
    prob_random = self.task_params.noise_model.prob_random
    action_status = [] 
    for i in range(batch_size):
      action = actions[i]*1
      state = states[i]*1
      rng = episodes[i].rng_noise
      u = rng.rand()
      status = True
      if u < prob_random and action == 3: 
        action = 0
        status = False
      
      if action == 3:
        _ = state
        for k in range(1):
          nn = self.task.graph.get_neighbours([_])
          __ = nn[0, 3]
          if __ == -1:
            break
          _ = __
        out_state = _
      elif action == 0:
        out_state = state
      else:
        nn = self.task.graph.get_neighbours([state])
        out_state = nn[0, action]
        if False:
          nn = self.task.graph.get_neighbours([out_state])
          out_state = nn[0, 3] if nn[0, 3] != -1 else out_state

      assert(out_state != -1)
      out_states.append(out_state)
      if not sim:
        episodes[i].states.append(out_state*1.)
        episodes[i].executed_actions.append(actions[i]*1.)
        episodes[i].action_status.append(status)
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
      views = self.render_views(states)
      # ATTENTION. Len of Views will always be 1
      for i in range(len(views)):
        if self.task_params.perturb_views:
          views[i] = perturb_images(views[i], [e.rng_noise for e in self.episodes], 0.1)
          # self.task_params.noise_model.prob_random)

      for i, _ in enumerate(views):
        outputs['views_{:d}'.format(i)] = views[i]
    
    if to_output.top_view_roads:
      roads = mu.generate_egocentric_maps(self.task.scaled_roads,
        self.task_params.view_scales, self.task_params.fovs, loc, x_axis, y_axis)
      for i, _ in enumerate(roads):
        outputs['roads_{:d}'.format(i)] = np.expand_dims(roads[i], -1)

    if to_output.loc_on_map:
      for i, sc in enumerate(self.task_params.view_scales):
        outputs['loc_on_map_{:d}'.format(i)] = np.concatenate((loc*sc, theta), axis=1)

    outputs['views_xyt'] = np_states 

    # if to_output.executed_action:
    return outputs


class TopViewDiscreteEnv(DiscreteEnv):
  """Observation is the top-view of the environment.
  Actions are simple grid world actions.
    - Rotate left, right, move straight stay in place.
    - With some probability it stays in place.
  """
  def __init__(self, name, dataset, flip, task_params, road=None, view=None, rng=None, r_obj=None):
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
    
    assert(self.task_params.ignore_roads == True)
    self._setup_noise()
    self._compute_graph()
    self._preprocess_for_task()

  def _preprocess_for_task(self):
    # Resize views with antialiasing.
    self.task.scaled_views = mu.resize_maps(self.task.view,
      self.task_params.view_scales, 'antialiasing')
    self.task.scaled_roads = mu.resize_maps((self.task.road*255).astype(np.uint8),
      self.task_params.view_scales, 'antialiasing')
    self.task.view = None #Takes a lot of memory so remove if not needed.
    self.task.road = None 
  
  def render_views(self, states):
    # Renders out the view from the state.
    np_states = np.array(states)
    loc, x_axis, y_axis, theta = self.get_loc_axis(np_states)
    views = mu.generate_egocentric_maps(self.task.scaled_views,
      self.task_params.view_scales, self.task_params.fovs, loc, x_axis, y_axis)
    return views

# def get_follower_task_params(batch_size=4, gt_delta_to_goal=False,
#   terminate_on_complete=False, compute_optimal_actions=False,
#   compute_optimal_actions_steps=2, gamma=0.99, add_flips=True,
#   max_dist=200, min_dist=25, act_f=[0., 2., 4., 8., 16.],
#   act_r=[15., 30., 45., 90.], history=0, teacher_steps=20,
#   rejection_sampling=False):
#   rl_params = utils.Foo(dense_wt=1.0, complete_reward=1.0, time_penalty=-0.01,
#     dist_thresh=25., gamma=gamma, terminate_on_complete=terminate_on_complete)
#   task_params = utils.Foo(min_dist=min_dist, max_dist=max_dist,
#     batch_size=batch_size, act_f=act_f, act_r=act_r, rl_params=rl_params,
#     gt_delta_to_goal=gt_delta_to_goal, compute_optimal_actions_step_size=0.9,
#     compute_optimal_actions_steps=compute_optimal_actions_steps,
#     compute_optimal_actions=compute_optimal_actions, add_flips=add_flips,
#     history=history, teacher_steps=teacher_steps,
#     rejection_sampling=rejection_sampling)
#   return task_params

def get_follower_task_params(batch_size=4, max_dist=200, min_dist=25,
  num_waypoints=1, path_length=20, history=0, add_flips=True, typ='sp', 
  data_typ='demonstartion', mapping_samples=20, share_start=False, dist_type='traj_dists', 
  extent_samples=200, plan_type='opt', plan_path=None,task_typ='forward',replay_buffer=0,tf_distort=False,
  minQ=20,reuseCount=4,spacious=False,multi_act=False):
  assert(data_typ in ['mapping', 'demonstartion'])
  task_params = utils.Foo(min_dist=min_dist, max_dist=max_dist,
    batch_size=batch_size, num_waypoints=num_waypoints,
    path_length=path_length, history=history, add_flips=add_flips, typ=typ,
    data_typ=data_typ, mapping_samples=mapping_samples, share_start=share_start, 
    dist_type=dist_type, extent_samples=extent_samples, plan_type=plan_type,
    plan_path=plan_path,task_typ=task_typ,replay_buffer=replay_buffer,tf_distort=tf_distort,
    minQ=minQ,reuseCount=reuseCount,spacious=spacious,multi_act=multi_act)
  assert(typ in ['sp', 'U'])
  assert(task_typ in ['return', 'forward'])
  return task_params

class Follower():
  """Provides data for the follower leader style problem. The leader generates
  a target trajectory, and outputs the set of images seen in going from the
  starting location to the goal location. Follower has noisy actuators and has
  to be able to get to the goal location the picked out by the leader."""
  
  def __init__(self, env, task_params):
    self.task = utils.Foo()
    self.task.env = env
    self.task_params = task_params
    
    if self.task_params.data_typ == 'mapping':
      # Initialize a mapper that can be used to generate images for the mapping
      # part of things.
      from env import mapper_env
      mapper_task_params = mapper_env.get_mapper_task_params(
        batch_size=task_params.batch_size, num_samples=task_params.mapping_samples, 
        extent_samples=task_params.extent_samples, 
        add_flips=task_params.add_flips, mapper_noise=0., 
        output_optimal_actions=True)
      self.task.env_mapper = mapper_env.MapperPlannerEnv(env, mapper_task_params)
  
  def reset(self, rng):
    rng_ = rng
    actions, waypoints, path_dists, traj_dists, goal_dists, paths = [], [], [], [], [], []
    task_params = self.task_params
    batch_size = self.task_params.batch_size
    spacious=task_params.spacious
    #print(self.task.env.task_params.t_prob_noise)
    # Generates the problem.
    start_id = None
    #print(batch_size)
    goal_point=self.task.env._sample_point_on_map(np.random.RandomState(rng.randint(1e6)),in_room=True,spacious=spacious)
    goal_dist=self.task.env.exp_nav._compute_distance_field(goal_point)
    path_length=self.task_params.path_length
    ### how do we pick interesting goals
    for i in range(batch_size):
      start_id = start_id if task_params.share_start else None
      #if self.task_params.typ == 'sp':
      #ATTENTION
      path,act=None,None
      l2_dist=1.0
      geo_dist=1.00
      count_attempt=0
      while(path is None):
        if (count_attempt > 200):
          goal_point=self.task.env._sample_point_on_map(np.random.RandomState(rng.randint(1e6),in_room=True,spacious=spacious))
          goal_dist=self.task.env.exp_nav._compute_distance_field(goal_point)
          count_attempt=0

        count_attempt+=1
        start_id=self.task.env._sample_point_on_map(np.random.RandomState(rng.randint(1e6)),spacious=spacious)
        geo_dist=goal_dist[int(start_id[0]),int(start_id[1])]
        l2_dist=((start_id[0]-goal_point[0])**2 + (start_id[1]-goal_point[1])**2)**0.5
        #print(geo_dist,l2_dist,self.task.env.task.step_size,path_length)
        if (count_attempt > 100 or \
                (geo_dist/self.task.env.task.step_size > 0.8*float(path_length) \
                and geo_dist < 1000 and l2_dist/geo_dist < 0.8)):
          path,act=self.task.env.exp_nav._find_shortest_path([start_id],goal_point,goal_dist,path_length+1,
                  noise_prob=self.task.env.task_params.t_prob_noise,rng=np.random.RandomState(rng.randint(1e6)),spacious=spacious)
      
      #_,path_t_noise=self.task.env.exp_nav._virtual_steps(act,path[0],goal_dist,noise=0.0,check_collision=False)
      #path=[path[0]]+path_t_noise
      #print(len(path),len(act),act[-1])
      #waypoint, path = graph.sample_random_path_waypoints(
      #  rng_, num_waypoints=task_params.num_waypoints,
      #  min_dist=task_params.min_dist, max_dist=task_params.max_dist,
      #  path_length=task_params.path_length, start_id=start_id)
      #start_id = waypoint[0]
      #traj_dist = graph.get_trajectory_distance(path.tolist(), 0.1, 
      traj_dist=0
      #  max_dist=task_params.path_length*4)
      #goal_dist = self.task.env.exp_nav.goal_dist
      # FIXME: goal_dist = graph.get_path_distance(path[-1:]) 
      action = deepcopy(act)
      # path_dist = graph.get_path_distance(path.tolist())
      
      waypoints.append(deepcopy(path))
      traj_dists.append(deepcopy(traj_dist))
      goal_dists.append(deepcopy(goal_dist))
      paths.append(deepcopy(path))
      #print(np.array(path).shape)
      #print(path)
      actions.append(deepcopy(action))
      # path_dists.append(path_dist)
    
    task = self.task
    task.paths, task.actions, task.traj_dists, task.waypoints, task.goal_dists = \
      paths, actions, traj_dists, waypoints, goal_dists
    # task.path_dists = path_dists
    task.history_f = [] 
    completed = [False for _ in paths]
    task.completed = completed

    # Set up problem for the follower agent. 
    init_states = [x[0] for x in paths]
    task.init_states = init_states
    env = self.task.env
    _ = env.reset(np.random.RandomState(rng.randint(1e6)), init_states=init_states, batch_size=batch_size,spacious=spacious)
    if self.task_params.data_typ == 'mapping':
      task.mapper_rng = utils.copy_rng(rng)
    return init_states

  def gen_data(self, rng):
    """IGNORE"""
    # This function is used to generate the same data that is used for
    # generating the problems for trajectory following, but this generates it
    # for mapping and planning.
    init_states = self.reset(rng)
    outputs = self._get_planning_data()
    # Benchmark the optimal actions to see if they are effective.
    # import pdb; pdb.set_trace()
    # num_steps = self.task_params.path_length
    # self.task.env_mapper.execute_actions_1(outputs['gt_actions'], init_states, 
    #   self.task.goal_dists, self.task.goal_nn, num_steps)
    # d_starts, d_ends = self.task.env_mapper.execute_actions(outputs['gt_actions'])
    
    # Cross check if things come back in the same coodinate frame etc.
    # _all_actions, _state_dist, _state_theta, _state_rel_orient = \
    #   self.task.env_mapper.decode_plan(outputs['gt_actions'], 
    #   self.task.init_states, self.task.goal_dists, self.task.goal_nn, 20)
    # _state_dist = _state_dist*5
    # oo = self._get_demonstration()
    return outputs

  def _get_planning_data(self):
    """IGNORE"""
    """Generate data for training the path planner.
    1. Call the _get_mapping_data function
    2. Also compute ground truth for planning problems."""
    init_states = self.task.init_states
    graph = self.task.env.task.graph
    outputs, _ = self._get_mapping_data()
    for s in init_states:
      assert(s == init_states[0]), \
        'init_states are not all the same {:s}.'.format(str(init_states))
    goal_dists = self.task.goal_dists
    goal_nn = self.task.env_mapper._get_node_nn(init_states[0])
    all_oas = []
    goal_imgs = []
    
    for i in range(len(init_states)):
      _ = goal_nn[goal_nn > -1]
      _n, _d = graph.get_action_distance_vec(goal_dists[i], _)
      _oa = _d == np.min(_d, 1)[:,np.newaxis]
      oas = np.zeros((goal_nn.shape[0], goal_nn.shape[1], goal_nn.shape[2], 4), dtype=np.bool)
      goal_img = np.ones((goal_nn.shape[0], goal_nn.shape[1], goal_nn.shape[2]), dtype=np.float)*np.inf
      goal_img[goal_nn>-1] = _d[:,0]
      goal_img = goal_img == 0
      goal_imgs.append(goal_img)
      oas[goal_nn > -1, :] = _oa
      all_oas.append(oas);
    
    goal_imgs = np.array(goal_imgs)
    goal_imgs = goal_imgs[:,::-1,:,:]
    goal_imgs = np.transpose(goal_imgs, [0,2,1,3])
    
    all_oas = np.array(all_oas)
    all_oas = all_oas[:,::-1,:,:,:]
    all_oas = np.transpose(all_oas, [0,2,1,3,4])
    all_oas = all_oas.astype(np.float32)
    
    goal_nn = goal_nn[::-1,:,:]
    goal_nn = np.transpose(goal_nn, [1,0,2])
    
    valid_mask = goal_nn[:,:,:1] > -1
    valid_mask = np.logical_or(np.zeros((len(init_states),1,1,1), dtype=np.bool), 
      valid_mask[np.newaxis,...])

    outputs['valid'] = valid_mask.astype(np.float32)
    outputs['gt_actions'] = all_oas
    outputs['goal_imgs'] = goal_imgs.astype(np.float32)
    self.task.planning = utils.Foo(goal_nn=goal_nn, gt_actions=all_oas)
    return outputs

  def execute_actions(self, action_volume, output_dir, global_step):
    """Given the action volume, executes the actions in open loop on the
    episode once to save the trajectories."""
    """IGNORE"""
    # Unroll the trajectory from the start state.
    num_steps = self.task_params.path_length
    d_starts, d_ends = self.task.env_mapper.execute_actions_1(action_volume, 
      self.task.init_states, self.task.goal_dists, self.task.planning.goal_nn,
      num_steps)
    
    # For debugging.
    # _all_actions, _state_dist, _state_theta, _state_rel_orient = \
    #   self.task.env_mapper.decode_plan(self.task.planning.gt_actions, 
    #     self.task.init_states, self.task.goal_dists, 
    #     self.task.planning.goal_nn, num_steps)
    # _state_dist = _state_dist*5
    
    _all_actions, _state_dist, _state_theta, _state_rel_orient = \
      self.task.env_mapper.decode_plan(action_volume, self.task.init_states, 
        self.task.goal_dists, self.task.planning.goal_nn, num_steps)
    _state_dist = _state_dist*5
    
    # Save mapping samples, init_states, goal_states
    if True:
      goal_states = [np.where(self.task.goal_dists[i] == 0)[0][0] 
        for i,_ in enumerate(self.task.init_states)]
      # goal_states = np.array(goal_states)
      # init_states = np.array(self.task.init_states)
      batch_data = utils.Foo(map_id_samples=self.task.map_id_samples,
        init_states=self.task.init_states, goal_states=goal_states,
        teacher_actions=_all_actions, teacher_dist=_state_dist,
        teacher_theta=_state_theta, teacher_rel_orient=_state_rel_orient, 
        batch=self.batch)

      out_dir = os.path.join(output_dir, 'plans', '{:08d}'.format(global_step))
      utils.mkdir_if_missing(out_dir)
      file_name = os.path.join(out_dir, '{:08d}.pkl'.format(self.batch))
      print(file_name)
      tt = vars(batch_data)
      utils.save_variables(file_name, tt.values(), tt.keys(), overwrite=True)
      # oo = self._get_demonstration()
      # Save data here for loading into the class for processing.
    return d_starts[:,np.newaxis], d_ends[:,np.newaxis], None
  
  def get_common_data(self):

    outputs = {}
    #print(self.task_params.data_typ)
    if self.task_params.task_typ=='return' and self.task_params.data_typ == 'mapping': 
      o1 = self._get_demonstration_flipped()
      # print([(k, v.dtype) for k, v in zip(o1.keys(), o1.values())])
      outputs.update(o1)

    if self.task_params.plan_type == 'opt':
      o1 = self._get_demonstration()
      # print([(k, v.dtype) for k, v in zip(o1.keys(), o1.values())])
      outputs.update(o1)

    elif self.task_params.plan_type == 'custom':
      o2 = self._get_demonstration_from_plan()
      # print([(k, v.dtype) for k, v in zip(o2.keys(), o2.values())])
      outputs.update(o2)

    return outputs

  def _get_demonstration_from_plan(self):
    """Load the data from file."""
    """Returns a demonstration of the trajectory that provides images and
    actions taken to convey the robot to the target location.
    Adds teacher_actions, teacher_xyt, teacher_dist, teacher_theta,
    teacher_rel_orient, teacher_views to the dictionary."""
    assert(self.task_params.plan_path is not None)
    file_name = os.path.join(self.task_params.plan_path, '{:08d}.pkl'.format(self.batch))
    tt = utils.load_variables(file_name)
    logging.error('%s', file_name)
    goal_states = [np.where(self.task.goal_dists[i] == 0)[0][0] 
      for i,_ in enumerate(self.task.init_states)]
    
    assert(np.allclose(np.array(self.task.init_states), tt['init_states']))
    assert(np.allclose(np.array(goal_states), tt['goal_states']))
    # assert(np.allclose(self.task.map_id_samples, tt['map_id_samples']))

    outputs = {}
    for k in ['teacher_actions', 'teacher_dist', 'teacher_theta', 'teacher_rel_orient']: 
      outputs[k] = tt[k]
    outputs['teacher_rel_orient'] = outputs['teacher_rel_orient']*1.
    outputs['teacher_views'] = np.zeros((8,20,224,224,3), dtype=np.uint8)
    outputs['teacher_xyt'] = np.zeros((8,20,3), dtype=np.int32)
    return outputs

  def _get_demonstration(self,name='teacher'):
    """Returns a demonstration of the trajectory that provides images and
    actions taken to convey the robot to the target location.
    Adds teacher_actions, teacher_xyt, teacher_dist, teacher_theta,
    teacher_rel_orient, teacher_views to the dictionary."""
    task = self.task
    task_params = self.task_params
    env = task.env
    outputs = {}
    
    teacher_actions = np.array(task.actions)
    teacher_states = np.array(task.paths)
    init_states = np.array([x[0] for x in task.paths])
    #print((teacher_states))
    #print(teacher_states[0][0])
    #print(teacher_states[0].shape)

    teacher_states = teacher_states[:,:-1]
    #print((teacher_states).shape)
    outputs[name+'_actions'] = teacher_actions
    outputs[name+'_xyt'] = teacher_states #env.task.node_embedding[teacher_states,:]
    
    # Teacher locations wrt the map.
    teacher_dist, teacher_theta, teacher_rel_orient = \
      env.get_relative_coordinates(teacher_states, init_states) 
    outputs[name+'_dist'] = teacher_dist * env.task.building.env.resolution
    outputs[name+'_theta'] = teacher_theta 
    outputs[name+'_rel_orient'] = teacher_rel_orient 

    # Render out the views.
    if self.task.env.task_params.outputs.top_view:
      intermediate_views = env.render_views(teacher_states.reshape([-1,3]))[0]
      sh = [teacher_states.shape[0], teacher_states.shape[1],
        intermediate_views.shape[1], intermediate_views.shape[2],
        intermediate_views.shape[3]]
      intermediate_views = np.reshape(intermediate_views, sh)
      outputs[name+'_views'] = intermediate_views.astype(np.uint8)
    return outputs


  def _get_demonstration_flipped(self):
    """Returns a demonstration of the trajectory that provides images and
    actions taken to convey the robot to the target location.
    Adds teacher_actions, teacher_xyt, teacher_dist, teacher_theta,
    teacher_rel_orient, teacher_views to the dictionary."""
    task = self.task
    task_params = self.task_params
    env = task.env
    outputs = {}
    inverted_paths=copy.deepcopy(task.paths)
    for i1 in range(len(inverted_paths)):
      for i2 in range(len(inverted_paths[i1])):
        inverted_paths[i1][i2][2]+=np.pi
    teacher_actions = np.array(task.actions)
    teacher_states = np.array(inverted_paths)
    init_states = np.array([x[0] for x in inverted_paths])

    #print((teacher_states))
    #print(teacher_states[0][0])
    #print(teacher_states[0].shape)

    teacher_states = teacher_states[:,:-1]
    #print((teacher_states).shape)
    outputs['mapping_actions'] = teacher_actions
    outputs['mapping_xyt'] = teacher_states #env.task.node_embedding[teacher_states,:]
    
    # Teacher locations wrt the map.
    teacher_dist, teacher_theta, teacher_rel_orient = \
      env.get_relative_coordinates(teacher_states, init_states) 
    outputs['mapping_dist'] = teacher_dist * env.task.building.env.resolution
    outputs['mapping_theta'] = teacher_theta 
    outputs['mapping_rel_orient'] = teacher_rel_orient 

    # Render out the views.
    if self.task.env.task_params.outputs.top_view:
      intermediate_views = env.render_views(teacher_states.reshape([-1,3]))[0]
      sh = [teacher_states.shape[0], teacher_states.shape[1],
        intermediate_views.shape[1], intermediate_views.shape[2],
        intermediate_views.shape[3]]
      intermediate_views = np.reshape(intermediate_views, sh)
      outputs['mapping_views'] = intermediate_views.astype(np.uint8)
    return outputs

  def _get_mapping_data(self):
    """Returns set of image, pose pairs around the current location of the
    agent that are going to be used for mapping. Calls appropriate functions
    from the mapper_env class."""
    task, task_params = self.task, self.task_params
    env, env_mapper = task.env, task.env_mapper
    
    init_states = [x[0] for x in task.paths] 
    rng = task.mapper_rng
    id_samples = env_mapper._sample_mapping_nodes(init_states, rng)
    outputs = env_mapper._gen_mapping_data(id_samples, init_states)
    
    # FIXME?: Rename things in outputs.
    if task_params.share_start:
      # Check if the starting point is the same or not.
      for s in init_states:
        assert(s == init_states[0]), \
          'init_states are not all the same {:s}.'.format(str(init_states))
      for k in outputs.keys(): 
        outputs[k] = outputs[k][:1,...]
      id_samples[1:,:] = id_samples[:1,:]
    self.task.map_id_samples = id_samples
    return outputs, id_samples
 
  def pre_common_data(self, inputs):
    """Pre-computes the common data."""
    return inputs

  def pre_features(self, f):
    """Pre-computes the features."""
    return f
  
  def get_features(self, states, step_number=None):
    """Computes tensors that get fed into tensorflow at each time step."""
    task = self.task; env = task.env; task_params = self.task_params
    f = env.get_features(states)
    while len(task.history_f) < task_params.history:
      task.history_f.insert(0, copy.deepcopy(f))
    # Insert the latest frame.
    task.history_f.insert(0, copy.deepcopy(f))
    
    if self.task.env.task_params.outputs.top_view:
      view = np.concatenate([np.expand_dims(x['views_0'], -1) for x in task.history_f], -1)
      view = np.expand_dims(view, axis=1)
      f['view'] = view
      f.pop('views_0', None); 

    if self.task.env.task_params.outputs.top_view_roads:
      road = np.concatenate([np.expand_dims(x['roads_0'], -1) for x in task.history_f], -1)
      road = np.expand_dims(road, axis=1)
      f['road'] = road
      f.pop('roads_0', None); 
    
    f['loc_on_map'] = np.expand_dims(f['loc_on_map_0'], axis=1)
    f.pop('loc_on_map_0', None);

    f['view_xyt'] = np.expand_dims(f['views_xyt'], axis=1)
    
    # Compute distance from trajectory from current state.
    gt_dist = np.array([getattr(task, task_params.dist_type)[i][int(x[0])][int(x[1])] for i,x in enumerate(states)], 
      dtype=np.float32)
    #print(gt_dist.shape)
    f['gt_dist'] = np.reshape(gt_dist*1., [-1,1,1])

    cd1s, cd2s, _ = self._comptue_chamfer_distance()
    f['cd_prec'] = cd1s
    f['cd_recall'] = cd2s
    # Compute chamfer distance between trajectories.
    task.history_f.pop()
    return f

  def _comptue_chamfer_distance(self):
    episodes = self.task.env.episodes
    teacher_states = self.task.paths
    #nodes = self.task.env.task.graph.nodes
    cd1s = []; cd2s = [];
    for i in range(len(teacher_states)):
      teacher_traj = np.array(teacher_states[i])[0:2]*1.
      student_traj = np.array(episodes[i].states)[0:2]*1.
      tt = np.expand_dims(teacher_traj, 1) - np.expand_dims(student_traj, 0)
      tt = np.sqrt(np.sum(tt**2, 2)) / self.task.env.task_params.step_size
      cd1s.append(np.mean(np.min(tt,0)))
      cd2s.append(np.mean(np.min(tt,1)))
    cd1s = np.expand_dims(np.array(cd1s), 1)
    cd1s = np.expand_dims(np.array(cd1s), 2)
    cd2s = np.expand_dims(np.array(cd2s), 1)
    cd2s = np.expand_dims(np.array(cd2s), 2)
    
    task = self.task; task_params = self.task_params;
    episodes = self.task.env.episodes
    states = [(e.states[-1]) for e in episodes]
    gt_dist = [getattr(task, task_params.dist_type)[i][int(x[0])][int(x[1])] for i, x in enumerate(states)]
    gt_dist = np.array(gt_dist, dtype=np.float32)
    gt_dist = np.reshape(gt_dist, [-1,1,1])
    return cd1s, cd2s, gt_dist

  def take_action(self, states, actions, step_number=None):
    """Given states, executes actions. Returns the reward for each time step.
    """
    new_states = self.task.env.take_action(states, actions)
    rewards = [0 for _ in states]
    return new_states, rewards

  def get_optimal_action(self, states, j):
    """Is used to execute actions that an expert would have taken.
    Input:
        states: Whatever reset returns TODO.
    Output:
        acts is one-hot encoding of optimal action from states.
    """
    task = self.task; env = task.env; task_params = self.task_params
    acts = []
    for i in range(task_params.batch_size):
      d, n = env.exp_nav.find_best_action_set(states[i],getattr(task, task_params.dist_type)[i]\
              ,spacious=task_params.spacious,multi_act=task_params.multi_act)
      a=np.zeros([4])
      if task_params.multi_act:
        for d_i in d:
          a[int(d_i[0])]=1
      else:
        a[int(d[0])]=1
      #print(a)
      acts.append(a)
    acts = np.array(acts)*1
    self.get_opt_act=acts
    return acts 

  def get_targets(self, states, j):
    """Used to compute ground truth for things that the network should produce
    at each time step.
    gt_action: probability of taking each action
    gt_q_value: q-value for different actions
    """
    task = self.task; env = task.env; task_params = self.task_params
    # a = np.zeros((len(states), 1, self.task.actions.shape[0]), dtype=np.int32);
    # a[:,:,0] = 1;
    #a = self.get_optimal_action(states, j)
    assert(self.get_opt_act is not None)
    a=self.get_opt_act
    self.get_opt_act=None 
    ds = []
    for i in range(self.task_params.batch_size):
      #n, d = self.task.env.task.graph.get_action_distance(getattr(task, task_params.dist_type)[i], states[i])
      #d[n == -1] = d[0]*1.+1
      d=np.array([0,0,0,0])
      ds.append(np.reshape(d, [1,-1]))
    ds = np.concatenate(ds, 0) 
    ds = -1.*ds
    a = np.expand_dims(a, axis=1)*1
    ds = np.expand_dims(ds, axis=1)*1
    return {'gt_action': a, 'gt_q_value': ds}

  def get_gamma(self):
    return 0.99 #self.task_params.rl_params.gamma
  
  def make_vis_paper_wts(self, out_dir, suffix='', prefix='', pointer=None,
    map_wt=None, rel_pose_teacher_mapping=None, mapping_view=None, teacher_views=None):
    """ Visualizes the best reference view for each location."""
    # Find the best view for each thing and write down the thingy.
    bs, ts = map_wt.shape[:2]
    ind = np.argmax(map_wt, 2)
    theta = np.mod(np.round(np.arctan2(rel_pose_teacher_mapping[...,3], rel_pose_teacher_mapping[...,2])/np.pi*2.), 4)
    for i in range(bs):
      for t in range(ts):
        fig, _, axes = utils.subplot2(plt, (1,2), (5,5))
        ax = axes.pop()
        ax.imshow(teacher_views[i,t,:,:,:].astype(np.uint8))
        ax.axis('off')
        ax = axes.pop()
        ax.imshow(mapping_view[0,ind[i,t],:,:,:].astype(np.uint8))
        ax.axis('off')
        _p = rel_pose_teacher_mapping[i,t,ind[i,t],:]
        ax.set_title('({:.0f}, {:.0f}, {:.0f}$^\circ$)'.format(round(_p[0]), round(_p[1]), 90*theta[i,t,ind[i,t]]))
        out_file_name = os.path.join(out_dir, 'corres_vis', 
          '{:s}corres_vis{:s}_{:02d}_{:02d}.png'.format(prefix, suffix, i, t))
        fig.savefig(out_file_name, bbox_inches='tight')
        plt.close(fig)

  def make_vis_video(self, out_dir, suffix, prefix, view, teacher_views,
    mapping_view, rel_pose_teacher_mapping, pointer, map_wt):
    import matplotlib.animation as manimation
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Full Visualization', artist='matplotlib',
      comment='Visualization')
    fps = 2

    """Visualizes the optimal and executed trajectories, action failure and the
    steps."""
    match_ind = np.argmax(map_wt, 2)

    # Make a plot of the episode for environments in this batch.
    map_id_samples = self.task.map_id_samples
    
    ind = np.argmax(map_wt, 2)
    theta = np.mod(np.round(np.arctan2(rel_pose_teacher_mapping[...,3],
      rel_pose_teacher_mapping[...,2])/np.pi*2.), 4)

    full_view = self.task.env.task.scaled_views[0]
    vs = self.task.env.task_params.view_scales[0]
    step_size = self.task.env.task_params.step_size
    task = self.task
    env = task.env
    plt.style.use('fivethirtyeight')
    cm = utils.get_538_cm()

    def _get_trajectory_data(task, i):
      vs = task.env.task_params.view_scales[0]
      env = task.env
      optimal = task.paths[i]
      executed = env.episodes[i].states
      o_loc, _, _, o_theta = env.get_loc_axis(np.array(optimal).astype(np.int32))
      o_loc = o_loc*vs; 
      e_loc, _, _, e_theta = env.get_loc_axis(np.array(executed).astype(np.int32))
      e_loc = e_loc*vs
      action_status = np.array(env.episodes[i].action_status)
      map_id_samples = task.map_id_samples
      if map_id_samples is not None:
        m_loc, _, _, m_theta = env.get_loc_axis(np.array(map_id_samples[i,:]).astype(np.int32))
        m_loc = m_loc*vs
      return o_loc, e_loc, m_loc, m_theta, action_status
    
    def _adjust_size(ax, o_loc, e_loc):
      min_size = 12
      all_locs = np.concatenate([o_loc, e_loc], axis=0)
      min_ = np.min(all_locs, axis=0)
      max_ = np.max(all_locs, axis=0)
      mid_ = (min_+max_)/2.
      sz = np.maximum(1.2*np.max(max_-min_)/2., min_size)
      ax.set_xlim([mid_[0]-sz, mid_[0]+sz])
      ax.set_ylim([mid_[1]-sz, mid_[1]+sz])
      ax.get_xaxis().set_ticks([])
      ax.get_yaxis().set_ticks([])

    def _reset_figs(axes):
      for a in axes:
        a.clear()
        a.axis('off')
    
    matplotlib.rcParams['axes.titlesize'] = 8
    for i in range(self.task_params.batch_size):
      writer = FFMpegWriter(fps=fps, metadata=metadata)
      offset = 1
      o_loc, e_loc, m_loc, m_theta, action_status = _get_trajectory_data(task, i)
      pointer_i = np.concatenate([np.array(0)[np.newaxis], pointer[i,:]*1], 0)
      plt.style.use('fivethirtyeight')
      fig = plt.figure(figsize=(10,6.6))
      gs = gridspec.GridSpec(3,5)
      gs.update(left=0.0, right=1.0, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
      ax_view = plt.subplot(gs[:3,:3]) # First person view
      ax_teacher = plt.subplot(gs[0,3]) # Reference image
      ax_synth = plt.subplot(gs[0,4]) # Reference image
      # Location on map (low alpha for the whole trajectory and full alpha for parts traversed already)
      ax_map = plt.subplot(gs[-2:,-2:]) 
      
      out_file_name = os.path.join(out_dir, '{:s}env_vis{:s}_{:02d}.mp4'.format(prefix, suffix, i))
      with writer.saving(fig, out_file_name, 100):
        _reset_figs([ax_view, ax_teacher, ax_synth, ax_map])
        # Display the constant things with the map.
        map_legend_bbox_to_anchor = (0.0, 0.7)
        map_legend_loc = 'upper right'
        ref_imgs_label = 'Ref. Images    '
        ax_map.imshow(1-full_view[:,:,0].astype(np.float32)/255., 
          vmin=0., vmax=2.5, cmap='Greys', origin='lower')
        ax_map.imshow(full_view, alpha=0.6, origin='lower')
        _adjust_size(ax_map, o_loc, e_loc)
        ax_map.text(.5, 1., 'Overhead View (Visualization Only)',
          verticalalignment='top', horizontalalignment='center', transform=ax_map.transAxes,
          fontdict={'fontsize': 10, 'color': 'red'}, bbox=dict(facecolor='white', alpha=0.9, lw=0))
        writer.grab_frame(**{'facecolor':'black'})
        
        for k in range(m_loc.shape[0]):
          s = 4; t = m_theta[k,0]
          arrow = ax_map.arrow(m_loc[k,0], m_loc[k,1], s*np.cos(t), s*np.sin(t),
            head_width=2, head_length=2, fc='g', ec='g', alpha=0.8, width=.5)
        ref_img = arrow
        ax_map.legend([ref_img], [ref_imgs_label],
          loc=map_legend_loc, bbox_to_anchor=map_legend_bbox_to_anchor)
        writer.grab_frame(**{'facecolor':'black'})
        
        ax_map.plot(o_loc[0,0], o_loc[0,1], cm[0], alpha=1.0, ms=20, marker='.', ls='none', label='Start')
        handles, labels = ax_map.get_legend_handles_labels()
        ax_map.legend([ref_img]+handles, [ref_imgs_label]+labels,
          loc=map_legend_loc, bbox_to_anchor=map_legend_bbox_to_anchor)
        writer.grab_frame(**{'facecolor':'black'})
        
        ax_map.plot(o_loc[-1,0], o_loc[-1,1], cm[0], alpha=1.0, ms=20, marker='*', ls='none', label='Goal')
        handles, labels = ax_map.get_legend_handles_labels()
        ax_map.legend([ref_img] + handles, [ref_imgs_label]+labels, 
          loc=map_legend_loc, bbox_to_anchor=map_legend_bbox_to_anchor)
        writer.grab_frame(**{'facecolor':'black'})
        
        ax_map.plot(o_loc[:,0], o_loc[:,1], cm[0], alpha=0.5, label='Planned')
        handles, labels = ax_map.get_legend_handles_labels()
        ax_map.legend([ref_img] + handles, [ref_imgs_label]+labels, 
          loc=map_legend_loc, bbox_to_anchor=map_legend_bbox_to_anchor)
        writer.grab_frame(**{'facecolor':'black'})
        
        ax_map.plot(e_loc[:,0]-offset, e_loc[:,1]-offset, cm[1], alpha=0.5, label='Executed')
        handles, labels = ax_map.get_legend_handles_labels()
        ax_map.legend([ref_img] + handles, [ref_imgs_label]+labels, 
          loc=map_legend_loc, bbox_to_anchor=map_legend_bbox_to_anchor)
        writer.grab_frame(**{'facecolor':'black'})
        
        o_loc_handle = ax_map.plot([], [], cm[0], alpha=1.0)[0]
        o_loc_point_handle = ax_map.plot([], [], cm[0], label='Loc. on Plan', 
          marker='.', ms=12, alpha=1.0, ls='none')[0]
        e_loc_handle = ax_map.plot([], [], cm[1], alpha=1.0)[0]
        e_loc_point_handle = ax_map.plot([], [], cm[1], label='Actual Loc.', 
          marker='.', ms=12, alpha=1.0, ls='none')[0]
        failed_handle = ax_map.plot([], [], 'kx')[0]
        
        handles, labels = ax_map.get_legend_handles_labels()
        ax_map.legend([ref_img] + handles + [failed_handle], 
          [ref_imgs_label] + labels +['Noisy Acts ( 0)'],
          loc=map_legend_loc, bbox_to_anchor=map_legend_bbox_to_anchor)
        # ax_map.legend([ref_img] + handles + [failed_handle, rel_mem_handle], 
        #   [ref_imgs_label] + labels +['Noisy Actions (0)', 'Rel. Mem.'],
        #   loc=map_legend_loc, bbox_to_anchor=map_legend_bbox_to_anchor)
        
        ax_teacher.text(.5, 1., 'Actual Image on Path Plan\n(Vis Only)',
          verticalalignment='top', horizontalalignment='center', transform=ax_teacher.transAxes,
          fontdict={'fontsize': 10, 'color': 'red'}, bbox=dict(facecolor='white', alpha=0.9, lw=0))
        ax_synth.text(.5, 1., 'Relevant Visual Memory',
          verticalalignment='top', horizontalalignment='center', transform=ax_synth.transAxes,
          fontdict={'fontsize': 10, 'color': 'red'}, bbox=dict(facecolor='white', alpha=0.9, lw=0))
        ax_view.text(0., 1., "Robot's View",
          verticalalignment='top', horizontalalignment='left', transform=ax_view.transAxes,
          fontdict={'fontsize': 20, 'color': 'red'}, bbox=dict(facecolor='white', alpha=0.9, lw=0))
        view_text_handle = ax_view.text(1., 1., 't = {:2d}, $\eta_t$ = {:4.1f}'.format(0, 0),
          verticalalignment='top', horizontalalignment='right', transform=ax_view.transAxes,
          fontdict={'fontsize': 20, 'color': 'red'}, bbox=dict(facecolor='white', alpha=0.9, lw=0))
        writer.grab_frame(**{'facecolor':'black'})
        writer.grab_frame(**{'facecolor':'black'})
        logging.error('%d', action_status.shape[0])
        for j in range(1+action_status.shape[0]):
          # reset axes and figures
          p = int(np.round(pointer_i[j]))
          
          o_loc_handle.set_data(o_loc[:(p+1),0], o_loc[:(p+1),1])
          e_loc_handle.set_data(e_loc[:(j+1),0]-offset, e_loc[:(j+1),1]-offset)
          e_loc_point_handle.set_data(e_loc[j,0]-offset, e_loc[j,1]-offset)
          
          failed_actions = np.where(np.invert(action_status[:j]))[0]
          failed_handle.set_data(e_loc[failed_actions,0]-offset, e_loc[failed_actions,1]-offset) 
          rel_mem_handle = None
           
          if p <= match_ind.shape[1]-1:
            k = match_ind[i,p]; s = 6; t = m_theta[k,0];
            rel_mem_handle = ax_map.arrow(m_loc[k,0], m_loc[k,1], s*np.cos(t), s*np.sin(t),
              head_width=4, head_length=4, fc='m', ec='m', alpha=1.0, width=1.)
            ax_synth.imshow(mapping_view[0,match_ind[i,p],:,:,:].astype(np.uint8))
            ax_teacher.imshow(teacher_views[0,p,:,:,:].astype(np.uint8))
            o_loc_point_handle.set_data(o_loc[p,0], o_loc[p,1]) 
          else:
            ax_synth.imshow(np.zeros((1,1,3), dtype=np.uint8))
            ax_teacher.imshow(np.zeros((1,1,3), dtype=np.uint8))
          
          noise_action_str = 'Noisy Acts ({:2d})'.format(np.sum(np.invert(action_status[:j])))
          ax_map.legend([ref_img] + handles + [failed_handle, rel_mem_handle], 
            [ref_imgs_label] + labels +[noise_action_str, 'Rel. Mem.'],
            loc=map_legend_loc, bbox_to_anchor=map_legend_bbox_to_anchor)
          
          view_text_handle.set_text('t = {:2d}, $\eta_t$ = {:4.1f}'.format(j, pointer_i[j]))
          ax_view.imshow(view[i,j,:,:,:,0].astype(np.uint8))
          writer.grab_frame(**{'facecolor':'black'})
          if rel_mem_handle is not None: 
            rel_mem_handle.remove()
      plt.close()

  def make_vis_paper(self, out_dir, suffix='', prefix='', pointer=None, map_wt=None):
    """Visualizes the optimal and executed trajectories, action failure and the
    steps."""

    min_size = 12
    # Make a plot of the episode for environments in this batch.
    cd_prec, cd_recall, gt_dist = self._comptue_chamfer_distance()
    map_id_samples = self.task.map_id_samples

    full_view = self.task.env.task.scaled_views[0]
    vs = self.task.env.task_params.view_scales[0]
    step_size = self.task.env.task_params.step_size
    env = self.task.env
    plt.style.use('fivethirtyeight')
    
    for i in range(self.task_params.batch_size):
      fig = plt.figure(figsize=(6,10)); 
      gs = gridspec.GridSpec(5,3)
      ax = plt.subplot(gs[:3, :3]) 
      ax1 = plt.subplot(gs[3, :3])
      ax2 = plt.subplot(gs[4, :3])

      # Plot 1 with the trajectory on the map.
      ax.imshow(full_view, alpha=0.6, origin='lower')
     
      all_locs = []
      optimal = self.task.paths[i]
      executed = env.episodes[i].states
      offset = 1
      for j, (states, label) in enumerate(zip([optimal, executed], ['planned', 'executed'])): 
        loc, _, _, theta = env.get_loc_axis(np.array(states).astype(np.int32))
        loc = loc*vs; 
        loc = loc - j*offset
        ax.plot(loc[:,0], loc[:,1], label=label);
        all_locs.append(loc)
        if j == 0:
          ax.plot(loc[0,0], loc[0,1], 'm.', ms=20)
          ax.plot(loc[-1,0], loc[-1,1], 'm*', ms=20)
        if j == 1:
          # Plot where it got stuck.
          action_status = np.array(env.episodes[i].action_status)
          failed_actions = np.where(np.invert(action_status))[0]
          ax.plot(loc[failed_actions,0], loc[failed_actions,1], 'kx', label='failed')
      ax.legend()

      all_locs = np.concatenate(all_locs, axis=0)
      min_ = np.min(all_locs, axis=0)
      max_ = np.max(all_locs, axis=0)
      mid_ = (min_+max_)/2.
      sz = np.maximum(1.2*np.max(max_-min_)/2., min_size)
      ax.set_xlim([mid_[0]-sz, mid_[0]+sz])
      ax.set_ylim([mid_[1]-sz, mid_[1]+sz])
      ax.get_xaxis().set_ticks([])
      ax.get_yaxis().set_ticks([])
      if map_id_samples is not None:
        map_loc, _, _, theta = env.get_loc_axis(np.array(map_id_samples[i,:]).astype(np.int32))
        map_loc = map_loc*vs
        for k in range(map_loc.shape[0]):
          s = 4; t = theta[k,0]
          ax.arrow(map_loc[k,0], map_loc[k,1], s*np.cos(t), s*np.sin(t),
            head_width=2, head_length=2, fc='g', ec='g', alpha=0.2)
      
      # Plot 2 with the executed actions.
      ax = ax1
      teacher_actions = np.array(self.task.actions[i])
      t_ = ax.plot(teacher_actions, 'g.-', label='planned')
      executed_actions = np.array(env.episodes[i].executed_actions)
      e_ = ax.plot(executed_actions-0.2, 'b.-', alpha=0.5, label='executed')
      action_status = np.array(env.episodes[i].action_status)
      failed_actions = np.where(np.invert(action_status))[0]
      nn = len(teacher_actions)
      ax.plot(failed_actions, executed_actions[failed_actions]-0.2, 'kx', label='failed')
      ax.set_ylim([-0.25, 3.2])
      ax.get_yaxis().set_ticks([0,1,2,3])
      ax.get_yaxis().set_ticklabels(['Stay', 'Left', 'Right', 'Forward'])
      ax.get_xaxis().set_ticks(np.arange(0, nn, nn/10))
      ax.legend()
      ax.axhline(-0.2, color='k')
      
      ax = ax2
      pointer_i = pointer[i,:]*1
      pointer_i[1:] = pointer_i[1:] - pointer_i[:-1]
      ax.bar(np.arange(len(pointer_i)), height=pointer_i)
      ax.get_xaxis().set_ticks(np.arange(0, nn, nn/10))
      ax.set_ylabel('$\eta$')
      ax.axhline(0.0, color='k')
      # ax.axvline(-0.5, color='k')
      out_file_name = os.path.join(out_dir, '{:s}env_vis{:s}_{:02d}.png'.format(prefix, suffix, i))
      fig.savefig(out_file_name, bbox_inches='tight', transparent=True, pad_inches=0)
      plt.close(fig)
  
  def save_vis(self, out_dir, suffix='', prefix=''):
    student_states = np.concatenate([e.states for e in self.task.env.episodes], 0)[np.newaxis,:,:]
    teacher_states = np.concatenate(self.task.paths, 0)[np.newaxis,...]
    return [[student_states, teacher_states]]

  def make_vis(self, out_dir, suffix='', prefix=''):
    min_size = 12
    # Make a plot of the episode for environments in this batch.
    cd_prec, cd_recall, gt_dist = self._comptue_chamfer_distance()
    map_id_samples = None #self.task.map_id_samples

    fig, _, axes = utils.subplot2(plt, (2, self.task_params.batch_size), (5,5))
    full_view = self.task.env.task.scaled_views[0]
    vs = self.task.env.task_params.view_scales[0]
    step_size = self.task.env.task_params.step_size
    env = self.task.env
    full_view_file = os.path.join(out_dir, 'full_view.png')
    if not os.path.exists(full_view_file):
      cv2.imwrite(full_view_file, full_view)
    for i in range(self.task_params.batch_size):
      ax = axes.pop()
      # Plot 1 with the trajectory on the map.
      ax.imshow(full_view, alpha=0.6, origin='lower')
      if map_id_samples is not None:
        map_loc = env.get_loc_axis(np.array(map_id_samples[i,:]).astype(np.int32))[0]
        map_loc = map_loc*vs
        ax.plot(map_loc[:,0], map_loc[:,1], 'g*', alpha=0.5)
      
      all_locs = []
      for j, (states, cmap, sz, m, lw) in enumerate(zip(
        [self.task.paths[i], env.episodes[i].states], 
        ['copper', 'cool'], [40, 10], ['o', 'o'], [0, 0])):
        loc = env.get_loc_axis(np.array(states).astype(np.int32))[0]
        loc = loc*vs
        loc = loc[:,::-1]*1
        ax.scatter(loc[:,0], loc[:,1], c=np.arange(loc.shape[0]), s=sz,
          cmap=cmap, marker=m, edgecolor='k', lw=lw)
        all_locs.append(loc)
      all_locs = np.concatenate(all_locs, axis=0)
      min_ = np.min(all_locs, axis=0)
      max_ = np.max(all_locs, axis=0)
      mid_ = (min_+max_)/2.
      sz = np.maximum(1.2*np.max(max_-min_)/2., min_size)
      ax.set_xlim([mid_[0]-sz, mid_[0]+sz])
      ax.set_ylim([mid_[1]-sz, mid_[1]+sz])
      ax.get_xaxis().set_ticks([])
      ax.get_yaxis().set_ticks([])
      ax.set_title('pre: {:0.2f}, rec: {:0.2f}, dist: {:0.2f}'.format(
        cd_prec[i,0,0], cd_recall[i,0,0], gt_dist[i,0,0]))
      
      # Plot 2 with the executed actions.
      ax = axes.pop()
      teacher_actions = np.array(self.task.actions[i])
      t_ = ax.plot(teacher_actions, 'g.-', label='teacher')
      executed_actions = np.array(env.episodes[i].executed_actions)
      e_ = ax.plot(executed_actions-0.1, 'b.-', alpha=0.5, label='student')
      ax.set_ylim([-0.2, 3.2])
      ax.get_yaxis().set_ticks([0,1,2,3])
      
      ax.legend() #[t_, e_])
    out_file_name = os.path.join(out_dir, 
      '{:s}env_vis{:s}.png'.format(prefix, suffix))
    fig.savefig(out_file_name, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close(fig)

def test_follower_noise_continuous():
  from env import factory
  d = factory.get_dataset('campus', 'small')
  name = d.get_imset()[0]
  logging.error(name)
  top_view_param = get_top_view_discrete_env_task_params(
    prob_random=0.2, batch_size=4, map_max_size=200, step_size=1)
  e = TopViewDiscreteEnv(dataset=d, name='small', task_params=top_view_param,
    flip=False, rng=np.random.RandomState(0))
  follower_task_param = get_follower_task_params(batch_size=4, min_dist=4, 
    max_dist=20, path_length=40, num_waypoints=8)
  f = Follower(e, follower_task_param)
  rng = np.random.RandomState(0)
  init_states = f.reset(rng)
  f.get_common_data()
  states = init_states
  for i in range(20):
    feats = f.get_features(states)
    logging.error('%s', feats.keys())
    acts = f.get_optimal_action(states, 0)
    gt_actions = f.get_targets(states, 0)
    acts = np.argmax(acts, axis=1)
    states, reward = f.take_action(states, acts)
    logging.error('%s, %s', str(acts), str(states))
  f.make_vis('tmp', '_1000_test')


def test_follower():
  from env import factory
  d = factory.get_dataset('campus', 'small')
  name = d.get_imset()[0]
  logging.error(name)
  top_view_param = get_top_view_discrete_env_task_params(
    prob_random=0.2, batch_size=4, map_max_size=200, step_size=1)
  e = TopViewDiscreteEnv(dataset=d, name='small', task_params=top_view_param,
    flip=False, rng=np.random.RandomState(0))
  follower_task_param = get_follower_task_params(batch_size=4, min_dist=4, 
    max_dist=20, path_length=40, num_waypoints=8)
  f = Follower(e, follower_task_param)
  rng = np.random.RandomState(0)
  init_states = f.reset(rng)
  f.get_common_data()
  states = init_states
  for i in range(20):
    feats = f.get_features(states)
    logging.error('%s', feats.keys())
    acts = f.get_optimal_action(states, 0)
    gt_actions = f.get_targets(states, 0)
    acts = np.argmax(acts, axis=1)
    states, reward = f.take_action(states, acts)
    logging.error('%s, %s', str(acts), str(states))
  f.make_vis('tmp', '_1000_test')

def test_discrete_env():
  from env import factory
  d = factory.get_dataset('campus', 'small')
  name = d.get_imset()[0]
  logging.error(name)
  top_view_param = get_top_view_discrete_env_task_params(
    prob_random=0.1, batch_size=4, map_max_size=200, step_size=8)
  e = TopViewDiscreteEnv(dataset=d, name='small', task_params=top_view_param,
    flip=False, rng=np.random.RandomState(0))
  
  # Try to take random actions inside this thing.
  rng = np.random.RandomState(0)
  init_states = e.reset(rng)
  locs = []
  states = init_states
  for i in range(20):
    states = e.take_action(states, [3,3,3,3])
    loc, _, _, _ = e.get_loc_axis(states)
    locs.append(loc)
  print(np.array(locs)[:,0,:])

def test_discrete_env_noise():
  from env import factory
  import matplotlib.pyplot as plt
  fig, _, axes = utils.subplot2(plt, (1,4))
  
  d = factory.get_dataset('campus', 'small')
  name = d.get_imset()[0]
  logging.error(name)

  for n in [0., 0.1, 0.2, 0.5]:
    top_view_param = get_top_view_discrete_env_task_params(
      prob_random=n, batch_size=32, map_max_size=200)
    e = TopViewDiscreteEnv(dataset=d, name='small', task_params=top_view_param,
      flip=False, rng=np.random.RandomState(0))

    # Try to take random actions inside this thing.
    rng = np.random.RandomState(0)
    init_states = e.reset(rng)
    locs = []
    states = [init_states[0] for _ in init_states]
    # states = init_states
    actions = np.ones((20,), dtype=np.uint8)*3
    actions[5] = 1; actions[15] = 1
    
    for i in range(20):
      loc, _, _, _ = e.get_loc_axis(states)
      states = e.take_action(states, [actions[i]]*32)
      locs.append(loc)
    locs = np.array(locs)*1.
    
    # Plot all these different trajectories and see what they look like
    ax = axes.pop()
    logging.error('%s', str(locs.shape))
    print(locs[0,:,:])
    for l in range(locs.shape[1]):
      loc = locs[:,l,:]
      r = rng.randn()*0.001
      ax.plot(r + loc[:,0], r + loc[:,1])
      ax.plot(loc[0,0], loc[0,1], 'rx')
      ax.plot(loc[-1,0], loc[-1,1], 'gx')
    plt.savefig('/tmp/sgupta-tmp-a.png')
    plt.close(fig)

def test_follower_2():
  from env import factory
  d = factory.get_dataset('campus', 'mnist1')
  name = d.get_imset()[0]
  logging.error(name)
  top_view_param = get_top_view_discrete_env_task_params(
    prob_random=0.2, batch_size=4, view_scales=[0.125], fovs=[64],
    base_resolution=1.0, step_size=128, top_view=True)
  e = TopViewDiscreteEnv(dataset=d, name=name, task_params=top_view_param,
    flip=False, rng=np.random.RandomState(0))
  follower_task_param = get_follower_task_params(
    batch_size=4, min_dist=4, max_dist=20, path_length=40, 
    num_waypoints=8, typ='U')
  f = Follower(e, follower_task_param)
  rng = np.random.RandomState(0)
  init_states = f.reset(rng)
  common_data = f.get_common_data()
  states = init_states
  feats = []
  for i in range(80):
    feats.append(f.get_features(states))
    acts = f.get_optimal_action(states, 0)
    gt_actions = f.get_targets(states, 0)
    acts = np.argmax(acts, axis=1)
    states, reward = f.take_action(states, acts)
