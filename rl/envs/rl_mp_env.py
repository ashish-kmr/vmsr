from __future__ import print_function
import logging
import numpy as np, os, cv2 
import numpy.ma as ma
import skfmm
import matplotlib.pyplot as plt
from src import utils
from src import file_utils as fu
from src import map_utils as mu 
import copy
import re
import matplotlib.patches as patches
from env.mp_env import expert_navigator

make_map = mu.make_map
resize_maps = mu.resize_maps
compute_traversibility = mu.compute_traversibility
pick_largest_cc = mu.pick_largest_cc

def _get_relative_goal_loc(goal_loc, loc, theta):
  r = np.sqrt(np.sum(np.square(goal_loc - loc), axis=1))
  t = np.arctan2(goal_loc[:,1] - loc[:,1], goal_loc[:,0] - loc[:,0])
  t = t-theta[:,0] + np.pi/2
  return np.expand_dims(r,axis=1), np.expand_dims(t, axis=1)

class Building():
  def __init__(self, dataset, name, robot, env, flip=False, map=None):
    self.restrict_to_largest_cc = True
    self.robot = robot
    self.env = env

    # Load the building meta data.
    env_paths = dataset.load_building(name)
    materials_scale = 0.25
    self.materials_scale = materials_scale
    
    shapess = dataset.load_building_meshes(env_paths, 
      materials_scale=materials_scale)

    if env.make_y_into_z:
      for shapes in shapess:
        shapes.make_y_into_z()
    
    if flip: 
      for shapes in shapess: 
        shapes.flip_shape()

    vs = []
    for shapes in shapess:
      vs.append(shapes.get_vertices()[0])
    vs = np.concatenate(vs, axis=0)
    if map is None:
      map = make_map(env.padding, env.resolution, vertex=vs, sc=100.)
    map = compute_traversibility(
      map, robot.base, robot.height, robot.radius, env.valid_min,
      env.valid_max, env.num_point_threshold, shapess=shapess, sc=100.,
      n_samples_per_face=env.n_samples_per_face)

    self.env_paths = env_paths 
    self.shapess = shapess
    self.map = map
    self.traversable = map.traversible*1
    self.name = name 
    self.full_name = dataset.ver + '-' + name + '-' + 'flip{:d}'.format(flip)
    self.flipped = flip
    self.renderer_entitiy_ids = []
    if self.restrict_to_largest_cc:
      self.traversable = pick_largest_cc(self.traversable)
   
    
    free_xy = np.array(np.nonzero(self.traversable)).transpose()
    room_dims = self._get_room_dimensions(env_paths['room_dimension_file'], 
      env.resolution, map.origin, flip=flip)
    room_regex = '^((?!hallway).)*$'
    room_dims = self._filter_rooms(room_dims, room_regex)
    self.WC_dims = self._filter_rooms(room_dims, 'WC*')['dims']
    if len(self.WC_dims) > 0: self.WC_dims = self.WC_dims[:,[1,4,0,3]]
    self.room_dims = room_dims
    room_list = self._label_nodes_with_room_id(free_xy,room_dims)
    room_idx = (room_list>-1)
    self.free_room_xy = free_xy[room_idx[:,0]][:]
    if (self.free_room_xy.shape[0]==0):
      print('\n\n\nENV IS BAD',name,'\n\n\n')
    
    self.room_xy = copy.deepcopy(self.traversable)
    
    #blocked_xy=np.array(np.nonzero(~self.traversable)).transpose()
    x_l,y_l = np.meshgrid(range(self.room_xy.shape[0]),range(self.room_xy.shape[1]))
    x_l = x_l.reshape([-1,1])
    y_l = y_l.reshape([-1,1])
    blocked_xy = np.concatenate([x_l,y_l],axis=1)
    room_list_on_blocked = self._label_nodes_with_room_id(
      blocked_xy.astype(np.float32), room_dims, width=10)
    room_idx_blocked = (room_list_on_blocked>-1)
    idx_lst = blocked_xy[room_idx_blocked[:,0]]
    self.room_xy[idx_lst[:,0], idx_lst[:,1]] = True

  def _label_nodes_with_room_id(self, xyt, room_dims,width=0):
    # Label the room with the ID into things.
    node_room_id = -1*np.ones((xyt.shape[0], 1))
    dims = room_dims['dims']
    for x, name in enumerate(room_dims['names']):
      all_ = np.concatenate((xyt[:,[0]] > dims[x,1] + width,
                             xyt[:,[0]] < dims[x,4] - width,
                             xyt[:,[1]] > dims[x,0] + width,
                             xyt[:,[1]] < dims[x,3] - width), axis=1)
      node_room_id[np.all(all_, axis=1), 0] = x
    return node_room_id

  def _filter_rooms(self,room_dims, room_regex):
    pattern = re.compile(room_regex)
    ind = []
    for i, name in enumerate(room_dims['names']):
      if pattern.match(name):
        ind.append(i)
    new_room_dims = {}
    new_room_dims['names'] = [room_dims['names'][i] for i in ind]
    new_room_dims['dims'] = room_dims['dims'][ind,:]*1
    return new_room_dims

  def _get_room_dimensions(self, file_name, resolution, origin, flip=False):
    if fu.exists(file_name):
      a = utils.load_variables(file_name)['room_dimension']
      names = list(a.keys())
      dims = np.concatenate(list(a.values()), axis=0).reshape((-1,6))
      ind = np.argsort(names)
      dims = dims[ind,:]
      names = [names[x] for x in ind]
      if flip:
        dims_new = dims*1
        dims_new[:,1] = -dims[:,4]
        dims_new[:,4] = -dims[:,1]
        dims = dims_new*1

      dims = dims*100.
      dims[:,0] = dims[:,0] - origin[0]
      dims[:,1] = dims[:,1] - origin[1]
      dims[:,3] = dims[:,3] - origin[0]
      dims[:,4] = dims[:,4] - origin[1]
      dims = dims / resolution
      out = {'names': names, 'dims': dims}
      
    else:
      logging.error('Room information not found. File %s does not exist.', file_name)
      out = None
    return out

  def _vis_room_dimensions(self):
    dims = self.room_dims['dims']
    traversable = self.traversable
    assert(dims is not None)
    # Plot things here for visualization.
    fig,  _, axes = utils.subplot2(plt, (1,1), (10,10))
    ax = axes.pop()
    ax.imshow(traversable)
    for i in range(dims.shape[0]): 
      _p = patches.Rectangle(dims[i,:2], dims[i,3]-dims[i,0],
        dims[i,4]-dims[i,1], fill=False, color='red') 
      ax.add_patch(_p)
    out_file_name = os.path.join('tmp', 'vis-rooms', '{:s}.png'.format(self.full_name))
    plt.savefig(out_file_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

  def set_r_obj(self, r_obj):
    self.r_obj = r_obj

  def load_building_into_scene(self):
    assert(self.shapess is not None)
    
    # Loads the scene.
    self.renderer_entitiy_ids += self.r_obj.load_shapes(self.shapess, 
      'flipped{:d}'.format(self.flipped))
    # Free up memory, we dont need the mesh or the materials anymore.
    self.shapess = None
  
  def to_actual_xyt(self, pqr):
    """Converts from node array to location array on the map."""
    out = pqr*1.
    return out

  def set_building_visibility(self, visibility):
    self.r_obj.set_entity_visible(self.renderer_entitiy_ids, visibility)

  def render_views(self, nodes, perturbs=None):
    # List of nodes to render.
    # perturbs is [1 x 2], containing perturbs for height of camera, elevation angle.
    self.set_building_visibility(True)
    if perturbs is None:
      perturbs = np.zeros((len(nodes), 2), dtype=np.float32)
    assert(len(perturbs) == len(nodes))

    imgs = []
    r = 2

    for i in range(len(nodes)):
      elevation_z = r * np.tan(np.deg2rad(self.robot.camera_elevation_degree+perturbs[i][1]))
      xyt = nodes[i]
      lookat_theta = 3.0 * np.pi / 2.0 - xyt[2]
      nxy = np.array([xyt[0], xyt[1]]).reshape(1, -1)
      nxy = nxy * self.map.resolution
      nxy = nxy + self.map.origin
      camera_xyz = np.zeros((1, 3))
      camera_xyz[...] = [nxy[0, 0], nxy[0, 1], self.robot.sensor_height+perturbs[i][0]]
      camera_xyz = camera_xyz / 100.
      lookat_xyz = np.array([-r * np.sin(lookat_theta),
                             -r * np.cos(lookat_theta), elevation_z])
      lookat_xyz = lookat_xyz + camera_xyz[0, :]
      self.r_obj.position_camera(camera_xyz[0, :].tolist(), lookat_xyz.tolist(), 
        [0.0, 0.0, 1.0])
      img = self.r_obj.render(take_screenshot=True, output_type=0)
      img = [x for x in img if x is not None]
      img = np.concatenate(img, axis=2)#.astype(np.uint8)
      imgs.append(img)

    self.set_building_visibility(False)
    return imgs


def _arch_setup(self):
    args_ = [('ver', 'v0'), ('num_steps', 'ns20'), ('image_cnn', 'rs18'),
    ('freeze_conv', 'frz1'), ('batch_norm', 'bn1'), ('dim_reduce_neurons',
    'dr0'), ('sample_gt_prob', 'zero'), ('map_type', 'map1'), ('dnc', 'dnc2'),
    ('combine_type', 'wtadd'), ('map_source', 'samples'), ('aux_loss', 'aux0'), 
    ('anneal_gt_demon', 'zero'), ('bn_pose', 'bnp1'), ('increment_fn_type', 'sigmoid')] 
    return utils.DefArgs(args_)

def get_task_params_from_string(cfg_str):
  _args = [('batch_size', 'bs1'), ('step_size', 'sz8'), ('nori', 'o12'),
    ('min_goal_dist', '36'), ('max_goal_dist', '44'), ('success_thresh', '16'),
    ('time_penalty', 'n0x1'), ('num_goals', '1'), ('num_starts_per_goal', '1'),
    ('terminate_on_done', '1'), ('max_time_steps', '20'), ('reward_type', 'sparse'),
    ('imset', 'area3')]
  def_args = utils.DefArgs(_args)
  print('Default string: ', def_args.get_default_string())
  task_params = def_args.process_string(cfg_str)
  task_params.batch_size = int(task_params.batch_size[2:])
  task_params.step_size = int(task_params.step_size[2:])
  task_params.nori = int(task_params.nori[1:])
  task_params.time_penalty = utils.str_to_float(task_params.time_penalty)

  int_keys = ['min_goal_dist', 'max_goal_dist', 'success_thresh',
    'num_goals', 'num_starts_per_goal', 'terminate_on_done',
    'max_time_steps']
  for k in int_keys:
    setattr(task_params, k, int(getattr(task_params, k).replace('n', '-')))
  task_params.terminate_on_done = task_params.terminate_on_done > 0
  task_params = get_task_params(**vars(task_params))
  return task_params 

def get_task_params(batch_size=8, step_size=8., nori=12, min_goal_dist=36,
  max_goal_dist=44, success_thresh=16., time_penalty=-0.1, num_goals=1,
  num_starts_per_goal=1, terminate_on_done=True, max_time_steps=20, reward_type='sparse', imset='area3'):
  """
    max_time_steps: These many calls to step function will cause the done to be True.
  """
  assert(reward_type in ['sparse','sparse4','dense', 'dense2'])
  task_params = utils.Foo(batch_size=batch_size, step_size=step_size,
    nori=nori, max_goal_dist=max_goal_dist, min_goal_dist=min_goal_dist,
    num_goals=num_goals, num_starts_per_goal=num_starts_per_goal,
    time_penalty=time_penalty, success_thresh=success_thresh,
    terminate_on_done=terminate_on_done, max_time_steps=max_time_steps,
    reward_type=reward_type, imset=imset)
  return task_params

class EnvMultiplexer():
  # Samples an environment at each iteration.
  def __init__(self, imlist, dataset, task_params, r_obj):
    self.envs = []
    for imname in imlist:
      self.envs.append(MPEnv(imname, dataset, False, task_params, r_obj=r_obj)) 

  def reset(self, rng):
    self.env_id = rng.choice(len(self.envs))
    states = self.envs[self.env_id].reset(rng)
    return states

  def take_action(self, states, action, only_if_not_done=False):
    states, rewards, dones = self.envs[self.env_id].take_action(states, action, only_if_not_done=only_if_not_done)
    return states, rewards, dones
  
  def get_features(self, states):
    feats = self.envs[self.env_id].get_features(states)
    return feats

  def vis_top_view_i(self, i, log_dir, prefix, suffix): 
    return self.envs[self.env_id].vis_top_view_i(i, log_dir, prefix, suffix)
  
  def vis_fp_view_i(self, i, log_dir, prefix, suffix): 
    return self.envs[self.env_id].vis_fp_view_i(i, log_dir, prefix, suffix)

class MPEnv():
  """Observation is the first person view of the environment.  Actions are
  simple macro actions: Rotate left, right, move straight stay in place, except
  that the agent can end up in truncated gaussian neighborhoods.
  """
  def __init__(self, name, dataset, flip, task_params, task_type = 'GoToPos', rng=None, r_obj=None):
    self.dilation_cutoff = 4
    self.name = name
    self.rng = np.random.RandomState(0) if rng is None else rng
    self.task = utils.Foo()
    self.task_params = task_params
    self.task.building = dataset.load_data(name, flip=flip)
    
    # self.task.road = self.task.building.traversable
    # self.free_rooms = self.task.building.free_room_xy
    # self.free_xy = None
    #WC_mask = self._compute_distance_field_WC()
    
    self.r_obj = r_obj
    self.resolution = 5
    self.n_ori = task_params.nori
    self.angle_value = [0, 2.0*np.pi/self.n_ori, -2.0*np.pi/self.n_ori, 0]
    self.task.road = self.task.building.traversable
    self.room_map=self.task.building.traversable
    self.task.step_size = self.task_params.step_size
    self.exp_nav=expert_navigator(self)
   
    self._preprocess_for_task()
    self._generate_problems(self.rng, task_params.num_goals, task_params.num_starts_per_goal, task_type)
    
  def _map_to_point(self, x, y):
    r = self.resolution
    o = [0,0]
    x, y = x*r, y*r
    x, y = x + o[0], y + o[1]
    return x, y
  
  def _sample_point_on_map(self, rng):
    traversable = self.task.building.traversable
    free_xy = np.array(np.nonzero(traversable)).transpose()
    start_id = rng.choice(free_xy.shape[0])
    start_orientation = rng.rand()*2*np.pi
    start = free_xy[start_id]
    assert(traversable[start[0], start[1]])
    return list(start)+[start_orientation]
  
  def get_traversable_for_vis(self):
    """Return a traversable for visualization."""
    tt = self.task.building.traversable.astype(np.float32)
    return tt

  def _preprocess_for_task(self):
    building = self.task.building
    building.set_r_obj(self.r_obj)
    building.load_building_into_scene()
    if building.shapess is not None:
      building.shapess = None
    # Prepare scaled_views and scaled_roads for visualizations.
    # view = (self.task.road*255).astype(np.uint8)
    # view = np.expand_dims(view, 2)*np.ones((1,1,3), dtype=np.uint8)
    # self.task.view = view
    # self.task.scaled_views = resize_maps(self.task.view,
    #   self.task_params.view_scales, 'antialiasing')
    # self.task.scaled_roads = resize_maps((self.task.road*255).astype(np.uint8), 
    #   self.task_params.view_scales, 'antialiasing')

  def render_views(self, states, perturbs=None):
    building = self.task.building
    states_flipped = [[st[1], st[0], -st[2]+np.pi/2] for st in states]
    imgs = building.render_views(states_flipped, perturbs)
    views = np.array(imgs)
    return views

  def take_action(self, states, actions, only_if_not_done=False):
    out_states, rewards, dones = [], [], []
    for i in range(len(states)):
      if only_if_not_done:
        if not self.episodes[i].done:
          out_state, reward, done = self.take_action_i(i, states[i], actions[i])
        else:
          out_state = list(states[i]); reward = np.NaN; done = True;
      else:
        out_state, reward, done = self.take_action_i(i, states[i], actions[i])
      out_states.append(out_state)
      rewards.append(reward)
      dones.append(done)
    return out_states, rewards, dones

  def take_action_i(self, i, state, action):
    """
    Actions are discrete:
      [0 (stay in place), 1(turn left), 2(turn right), 3(straight ahead)].
    Episode must not have finished when calling this function.
    Checks if the episode has finished, either because it reached the goal
    location, or if the time is over.
    """
    traversable = self.task.building.traversable
    episode = self.episodes[i]
    boundary_limits = traversable.shape
    angle_value = self.angle_value
    assert(not episode.done)
    
    executed_action = action*1
    state = list(state)
    out_state = list(state)
    du = self.task_params.step_size
    if executed_action == 3:
      angl = out_state[2]
      out_state[0] = out_state[0] + np.cos(angl)*du
      out_state[1] = out_state[1] + np.sin(angl)*du
      out_state[2] = angl
    elif executed_action > 0:
      out_state[2] += angle_value[executed_action]
    
    action_will_succeed = np.all(np.array(out_state[0:2]) < np.array(boundary_limits)) 
    action_will_succeed = action_will_succeed and np.all(np.array(out_state[0:2]) >= np.array([0,0]))
    action_will_succeed = action_will_succeed and traversable[int(out_state[0]), int(out_state[1])] == True
    
    if not action_will_succeed:
      # The action failed for some reason.
      out_state = list(state)
      executed_action = 0 
    
    reward, done, dist = self.get_reward_i(i, out_state)
    done = done or episode.num_steps+1 == self.task_params.max_time_steps
    episode.states.append(list(out_state))
    episode.executed_actions.append(executed_action)
    episode.input_actions.append(action)
    episode.state_dists.append(dist)
    episode.rewards.append(reward)
    episode.done = done
    episode.num_steps += 1
    if done:
      # Update max reward for this test case.
      eprew = np.sum(np.array(episode.rewards))
      self.task.max_rewards[episode.task_id] = \
        np.maximum(eprew, self.task.max_rewards[episode.task_id])
    
    return out_state, reward, done

  def get_reward_i(self, i, state):
    done = False
    goal_dist = self.episodes[i].goal_dist[int(state[0]), int(state[1])] 
    success_thresh = self.task_params.success_thresh
    
    if self.task_params.reward_type == 'sparse':
      reward = self.task_params.time_penalty
    if self.task_params.reward_type == 'sparse4':
      reward = self.task_params.time_penalty
      if goal_dist>self.task_params.max_goal_dist/4:    reward = self.task_params.time_penalty
      else:
          rel_dist = goal_dist * 2. / self.task_params.max_goal_dist 
          rel_success_dist = success_thresh * 2. / self.task_params.max_goal_dist
          reward = self.task_params.time_penalty * (1-np.exp(rel_success_dist-np.maximum(rel_success_dist, rel_dist)))
    elif self.task_params.reward_type == 'dense':
      rel_dist = goal_dist / success_thresh
      reward = self.task_params.time_penalty * (1-np.exp(1-np.maximum(1, rel_dist)))
    elif self.task_params.reward_type == 'dense2':
      rel_dist = goal_dist * 2. / self.task_params.max_goal_dist 
      rel_success_dist = success_thresh * 2. / self.task_params.max_goal_dist
      reward = self.task_params.time_penalty * (1-np.exp(rel_success_dist-np.maximum(rel_success_dist, rel_dist)))

    if goal_dist <= success_thresh:
      reward += 1.
      done = True

    return reward, done, goal_dist
  
  def _compute_distance_field(self, goal):
    traversable = self.task.building.traversable
    masked_t = ma.masked_values(traversable*1, 0) 
    goal_x, goal_y = int(goal[0]), int(goal[1])
    masked_t[goal_x, goal_y] = 0 
    dd = skfmm.distance(masked_t, dx=1)
    dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan))) 
    dd = ma.filled(dd, np.max(dd)+1) 
    return dd

  def _compute_distance_field_WC(self, gradient = 1.0):
    goal_dims = self.task.building.WC_dims
    traversable = self.task.building.traversable
    masked_t = ma.masked_values(traversable*1, 0) 
    for goal_dim_i in goal_dims:
        goal_dim_i = [int(itr_i) for itr_i in goal_dim_i] 
        x0, x1, y0, y1 = goal_dim_i
        masked_t[x0:x1, y0:y1] = 0
    dd = skfmm.distance(masked_t, dx=gradient)
    dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan))) 
    dd = ma.filled(dd, np.max(dd)+1) 
    return dd
  
  def _sample_goal(self, rng, start_pos, min_goal_dist, max_goal_dist, task_type):
    if task_type == 'GoToPos':
        traversable = self.task.building.traversable
        dd = self._compute_distance_field(start_pos)
    elif task_type == 'Semantic':
        traversable = self.task.building.traversable
        dd = self._compute_distance_field_WC()

    good_xy = np.array(np.nonzero(np.logical_and(dd < max_goal_dist, dd >= min_goal_dist))).transpose()
    end_id = rng.choice(good_xy.shape[0])
    end_orientation = rng.rand()*2*np.pi
    end = good_xy[end_id]
    assert(traversable[end[0], end[1]])
    return list(end)+[end_orientation]

  def _generate_problems(self, rng, num_goals, num_starts_per_goal, task_type):
    # Generate problems that will get sampled from during training.
    goal_states = np.zeros([num_goals*num_starts_per_goal, 3], dtype=np.float32)
    start_states = np.zeros([num_goals*num_starts_per_goal, 3], dtype=np.float32)
    exp_nsteps_s = np.zeros([num_goals*num_starts_per_goal], dtype=np.float32)
    exp_nsteps = np.zeros([num_goals*num_starts_per_goal], dtype=np.float32)
    max_rewards = -np.Inf*np.ones([num_goals*num_starts_per_goal,], dtype=np.float32)
    goal_dists = [None for _ in range(goal_states.shape[0])]
    
    k = 0
    for i in range(num_goals):
      rng_i = utils.get_rng(rng)
      if task_type == 'GoToPos':
          goal_state = self._sample_point_on_map(rng)
          goal_dist = self._compute_distance_field(goal_state)
      elif task_type == 'Semantic': 
          gcoords = self.task.building.WC_dims[0]
          goal_state = [np.mean(gcoords[0:2]), np.mean(gcoords[2:4]), 0.0]
          goal_dist = self._compute_distance_field_WC()
      
      for j in range(num_starts_per_goal):
        rng_ij = utils.get_rng(rng_i)
        start_state = self._sample_goal(rng_ij, goal_state,
          self.task_params.min_goal_dist, self.task_params.max_goal_dist, task_type)
        exp_nstep_s = len(self.exp_nav._find_shortest_path([start_state], None, goal_dist, spacious = True)[1])
        exp_nstep = len(self.exp_nav._find_shortest_path([start_state], None, goal_dist, spacious = False)[1])
        goal_states[k,:] = goal_state
        start_states[k,:] = start_state
        exp_nsteps_s[k] = exp_nstep_s
        exp_nsteps[k] = exp_nstep
        goal_dists[k] = goal_dist
        k += 1
    self.task.goal_states = goal_states
    self.task.start_states = start_states
    self.task.goal_dists = goal_dists
    self.task.max_rewards = max_rewards
    self.task.exp_nsteps_s = exp_nsteps_s
    self.task.exp_nsteps = exp_nsteps

  def reset(self, rng):
    init_states = []
    self.episodes = [None for _ in range(self.task_params.batch_size)]
    for i in range(self.task_params.batch_size):
      init_state = self.reset_i(i, rng)
      init_states.append(init_state)
    return init_states

  def reset_i(self, i, rng):
    """Resets episode i."""
    rng_i = utils.get_rng(rng)
    task_id = rng_i.choice(self.task.goal_states.shape[0])
    start_state = self.task.start_states[task_id,:].tolist()
    goal_state = self.task.goal_states[task_id,:].tolist()
    goal_dist = self.task.goal_dists[task_id]*1.
    exp_nsteps_s = self.task.exp_nsteps_s[task_id]
    exp_nsteps = self.task.exp_nsteps[task_id]

    episode = utils.Foo(rng=rng_i, states=[start_state], executed_actions=[],
      action_status=[], goal_state=goal_state, input_actions=[], state_dists=[],
      goal_dist=goal_dist, done=False, rewards=[], num_steps=0, task_id=task_id,
      exp_nsteps=exp_nsteps, exp_nsteps_s=exp_nsteps_s)
    self.episodes[i] = episode
    return start_state

  def get_metrics_i(self, i):
    """Computes and returns various metrics that we need for plotting
    results."""
    dd = {}
    ks = ['goal_state', 'input_actions', 'executed_actions', 'states', 'state_dists']
    for k in ks:
      dd[k] = np.array(getattr(self.episodes[i], k))
    ks = ['num_steps', 'done']
    for k in ks:
      dd[k] = getattr(self.episodes[i], k)
    dd['successful'] = dd['state_dists'][-1] <= self.task_params.success_thresh
    dd['spl'] = self._compute_spl(dd)
    dd['successful'] = int(dd['successful']*1)
    dd['done'] = int(dd['done']*1)
    dd['num_collisions'] = len(dd['input_actions']) - np.sum(np.array(dd['input_actions']) == np.array(dd['executed_actions']))
    dd['num_collisions'] = int(dd['num_collisions'])
    dd['max_task_reward'] = self.task.max_rewards[self.episodes[i].task_id]
    dd['avg_max_task_reward'] = np.mean(self.task.max_rewards)
    dd['avg_task_tried'] = np.mean(np.invert(np.isinf(self.task.max_rewards)))
    dd['avg_task_tried_reward'] = np.mean(self.task.max_rewards[np.invert(np.isinf(self.task.max_rewards))])
    dd['task_id'] = self.episodes[i].task_id
    dd['exp_nsteps_s'] = self.episodes[i].exp_nsteps_s
    dd['exp_nsteps'] = self.episodes[i].exp_nsteps
    return dd
    # Compute distance to goal at each time step.

  def _compute_spl(self, metrics):
    """Compute the SPL metric."""
    spl = 0
    if metrics['successful']:
      distance_travelled = np.sum(np.array(metrics['executed_actions']) == 3) * self.task_params.step_size
      spl = metrics['state_dists'][0] / np.maximum(metrics['state_dists'][0], distance_travelled)
      spl = float(spl)
    return spl
  
  def get_loc_axis(self, states):
    """Based on the node orientation returns X, and Y axis. Used to sample the
    map in egocentric coordinate frame.
    """
    loc = states[:,0:2]*1.
    theta = states[:,-1:]
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

  def _get_relative_goal_loc(self, goal_states, ref_states):
    goal_loc_xy = goal_states[:,:2][:,::-1]
    ref_loc_xy = ref_states[:,:2][:,::-1]
    ref_orient = ref_states[:,2]
    rr = np.sqrt(np.sum(np.square(goal_loc_xy - ref_loc_xy), axis=1))
    tt = np.arctan2(goal_loc_xy[:,1] - ref_loc_xy[:,1], goal_loc_xy[:,0] - ref_loc_xy[:,0])
    tt = tt + ref_orient - np.pi/2
    tt = np.mod(tt+np.pi, np.pi*2) - np.pi # Making it between -pi and pi
    
    rr = rr[:,np.newaxis]
    # Rescale distance using max_goal_dist, so that network inputs are well sized.
    rr = rr / self.task_params.max_goal_dist
    tt = tt[:,np.newaxis]
    rel_x = rr*np.cos(tt+np.pi/2)
    rel_y = rr*np.sin(tt+np.pi/2)
    goal_rt = np.concatenate([rr, tt], axis=1).astype(np.float32)
    goal_xy = np.concatenate([rel_x, rel_y], axis=1).astype(np.float32)
    goal_cossin = np.concatenate([rr, np.cos(tt), np.sin(tt)], axis=1).astype(np.float32)
    return goal_rt, goal_xy, goal_cossin


  def get_features(self, states):
    outputs = {}
    ref_states = np.array(states)
    goal_states = np.array([e.goal_state for e in self.episodes])
    goal_rt, goal_xy, goal_cossin = self._get_relative_goal_loc(goal_states, ref_states)
    time_step = np.array([e.num_steps for e in self.episodes]).astype(np.float32) / self.task_params.max_time_steps
    time_step = time_step[:,np.newaxis]

    views = self.render_views(states)
    outputs['view'] = views
    outputs['goal_rt'] = goal_rt
    outputs['goal_xy'] = goal_xy
    outputs['goal_rcossin'] = goal_cossin
    outputs['t'] = time_step 

    # if to_output.loc_on_map:
    #   for i, sc in enumerate(self.task_params.view_scales):
    #     outputs['loc_on_map_{:d}'.format(i)] = np.concatenate((loc*sc, theta), axis=1)
    # outputs['views_xyt'] = np_states 
    return outputs
    
  def vis_fp_view_i(self, i, log_dir, prefix='', suffix=''):
    # Make a visualization of the ith episode, using information till now in
    # the episode.
    min_size = 96
    states = self.episodes[i].states
   
    goal_rt, goal_xy, goal_cossin = \
      self._get_relative_goal_loc(np.array(self.episodes[i].goal_state)[np.newaxis,:], np.array(states))
    
    n = len(states)
    n_in_one_row = 4
    fig, _, axes = utils.subplot2(plt, (int(np.ceil(n*1./4)), 4), (5,5))
    views = self.render_views(states)
    executed_actions = np.array(self.episodes[i].executed_actions)
    input_actions = np.array(self.episodes[i].input_actions)
    for i in range(len(states)):
      ax = axes.pop()
      ax.imshow(views[i,:,:,:].astype(np.uint8))
      ax.set_axis_off()
      if i < len(states)-1:
        str_ = 'Executed: {:d}, Input: {:d}, theta: {:0.2f}'.format(executed_actions[i], input_actions[i], np.rad2deg(states[i][2]))
        str_ = str_ + '\n'
        str_ = str_ + 'rt: {:0.2f}, {:0.2f}. xy: {:0.2f}, {:0.2f}'.format(goal_rt[i,0], np.rad2deg(goal_rt[i,1]),
          goal_xy[i,0], goal_xy[i,1])
        ax.set_title(str_)
    
    out_file_name = os.path.join(log_dir, '{:s}env_vis_fp{:s}.png'.format(prefix, suffix))
    fig.savefig(out_file_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    I = cv2.imread(out_file_name)
    return I
 
  def vis_top_view_i(self, i, log_dir, prefix='', suffix=''):
    # Make a visualization of the ith episode, using information till now in
    # the episode.
    min_size = 96
    fig, _, axes = utils.subplot2(plt, (1, 3), (5,5))
    traversable = self.task.building.traversable
    # Get target goal location, and plot them.
    states = self.episodes[i].states
    states = np.array(states)
   
    goal_rt, goal_xy, goal_cossin = \
      self._get_relative_goal_loc(np.array(self.episodes[i].goal_state)[np.newaxis,:], np.array(states))
    
    # Plot the map
    ax = axes.pop()
    ax.imshow(traversable, alpha=0.6, origin='lower', cmap='gray', vmin=-0.5, vmax=1.5)
    # Plot the trajectory
    ax.plot(states[:,1], states[:,0], 'r.-', ms=10, alpha=0.7)
    # Plot the start location
    ax.plot(states[0,1], states[0,0], 'b.', ms=10, alpha=1.0)
    # Plot the goal location
    goal_state = self.episodes[i].goal_state
    ax.plot(goal_state[1], goal_state[0], 'g.', ms=800/fig.dpi, alpha=0.7)
    
    # Resize
    all_locs = np.concatenate((states, np.array(goal_state)[np.newaxis,:]), 0)
    min_ = np.min(all_locs, axis=0)
    max_ = np.max(all_locs, axis=0)
    mid_ = (min_+max_)/2.
    sz = np.maximum(1.2*np.max(max_-min_)/2., min_size)
    ax.set_ylim([mid_[0]-sz, mid_[0]+sz])
    ax.set_xlim([mid_[1]-sz, mid_[1]+sz])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax = axes.pop()
    executed_actions = np.array(self.episodes[i].executed_actions)
    ax.plot(executed_actions, 'r-', lw=4, ms=16, alpha=0.4, label='actually executed')
    input_actions = np.array(self.episodes[i].input_actions)
    ax.plot(input_actions+0.1, 'b-', lw=4, ms=16, alpha=0.4, label='executed')
    ax.grid(True); ax.set_ylim([-0.2, 3.2]);
    ax.set_xticks(np.arange(executed_actions.size))
    ax.legend()

    ax = axes.pop()
    ax.plot(goal_xy[:,0], goal_xy[:,1], 'r.')
    for i in range(goal_xy.shape[0]):
      ax.text(goal_xy[i,0], goal_xy[i,1], str(i))
    ax.axis('equal')

    out_file_name = os.path.join(log_dir, '{:s}env_vis_top{:s}.png'.format(prefix, suffix))
    fig.savefig(out_file_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    I = cv2.imread(out_file_name)
    return I

