from __future__ import print_function
import logging
import numpy as np, os, cv2, os, scipy, graph_tool as gt, skimage, itertools, copy
import matplotlib.pyplot as plt
from scipy import ndimage
from graph_tool import topology
from env import mp_env
from env import factory, noisy_nav_env
from src import utils
from src import rotation_utils as ru
from src import map_utils as mu 
from src import graph_utils as gu 
from src import graphs as graphs 
from env import toy_landmark_env as tle

generate_goal_images = mu.generate_goal_images

def _get_relative_goal_loc(goal_loc, loc, theta):
  r = np.sqrt(np.sum(np.square(goal_loc - loc), axis=1))
  t = np.arctan2(goal_loc[:,1] - loc[:,1], goal_loc[:,0] - loc[:,0])
  t = t-theta[:,0] + np.pi/2
  return np.expand_dims(r,axis=1), np.expand_dims(t, axis=1)

def get_cost_map_task_params(batch_size=4, max_dist=200, min_dist=25,
  history=0, add_flips=True, typ='rng', num_ref_points=0):
  task_params = utils.Foo(min_dist=min_dist, max_dist=max_dist,
    batch_size=batch_size, history=history, add_flips=add_flips, typ=typ,
    num_ref_points=num_ref_points)
  assert(typ in ['rng'])
  return task_params

class CostMap():
  """Provides data for learning the cost map."""
  def __init__(self, env, task_params):
    self.task = utils.Foo()
    self.task.env = env
    self.task_params = task_params
  
  def reset(self, rng):
    """Generate a new episode.
    - Sample a random start and goal location.
    - Compute the distance to goal from all location.
    """
    tp = self.task_params
    g = self.task.env.task.graph
    env = self.task.env
    task = self.task
    init_states, goal_states, dists, paths = [], [], [], []
    for i in range(tp.batch_size):
      s, e, path = g.sample_random_goal(rng, tp.min_dist, tp.max_dist)
      # Compute distance to goal from all nodes.
      dist = g.get_path_distance([e])
      # Compute atleast one path between the source and the goal (to sample
      # demonstrations from).
      
      init_states.append(s)
      goal_states.append(e)
      dists.append(dist)
      paths.append(path)
  
    task.init_states, task.goal_states, task.dists, task.paths = \
      init_states, goal_states, dists, paths
    task.history_f = []
    _ = env.reset(rng, init_states=init_states, batch_size=tp.batch_size)
    return init_states
  
  def _get_common_data(self, states):
    """Returns:
    - Free space map in the egocentric coordinate frame.
    - Goal location in the egocentric coordinate frame.
    """
    tp = self.task_params
    tp_ = self.task.env.task_params
    g = self.task.env.task.graph
    env = self.task.env
    task = self.task

    ff = env.get_features(states)

    loc, x_axis, y_axis, theta = env.get_loc_axis(states)
    goal_loc, _, _, _theta = env.get_loc_axis(task.goal_states)
    rel_goal_orientation = np.mod(np.int32((theta - _theta)/(np.pi/2)), 4)
    goal_dist, goal_theta = _get_relative_goal_loc(goal_loc, loc, theta)
    goal_imgs = generate_goal_images(tp_.view_scales, tp_.fovs, 4, 
      goal_dist, goal_theta, rel_goal_orientation)

    outputs = {}
    for i in range(len(tp_.fovs)):
      outputs['road_{:d}'.format(i)] = ff['roads_{:d}'.format(i)]*1.
      outputs['goal_img_{:d}'.format(i)] = goal_imgs[i]

    # Also return the (x,y,theta) locations of the intermediate states on the
    # path from start to goal.
    wpts = []
    num_ref_points = tp.num_ref_points
    for i in range(tp.batch_size):
      path = np.array(task.paths[i])
      rng = np.random.RandomState(0)
      ids = np.sort(rng.randint(len(path), size=num_ref_points))
      wpts.append(path[ids])
    wpts = np.array(wpts)
    
    waypoint_imgs = []
    for i in range(num_ref_points):
      _ = self._generate_state_images(states, wpts[:,i].tolist())
      _ = np.max(_[0], -1)
      waypoint_imgs.append(np.expand_dims(_, -1))
      outputs['ref_img_{:d}'.format(i)] = waypoint_imgs[-1]
    
    return outputs

  def _generate_state_images(self, current_states, goal_states):
    tp = self.task_params
    tp_ = self.task.env.task_params
    env = self.task.env
    loc, x_axis, y_axis, theta = env.get_loc_axis(current_states)
    goal_loc, _, _, _theta = env.get_loc_axis(goal_states)
    rel_goal_orientation = np.mod(np.int32((theta - _theta)/(np.pi/2)), 4)
    goal_dist, goal_theta = _get_relative_goal_loc(goal_loc, loc, theta)
    goal_imgs = generate_goal_images(tp_.view_scales, tp_.fovs, 4, 
      goal_dist, goal_theta, rel_goal_orientation)
    return goal_imgs

  def get_common_data(self):
    return self._get_common_data(self.task.init_states)

  def pre_common_data(self, inputs):
    return inputs

  def pre_features(self, f):
    return f
  
  def get_features(self, states, step_number=None):
    """ Current version does not return anything.
    """
    # Compute distance from trajectory from current state.
    f = {}
    # f = self._get_common_data(states)
    # f['road_0'] = np.expand_dims(f['road_0'], 1)
    # f['goal_img_0'] = np.expand_dims(f['goal_img_0'], 1)
    gt_dist = np.array([self.task.dists[i][x] for i, x in enumerate(states)],
      dtype=np.float32)
    f['gt_dist'] = np.reshape(gt_dist*1., [-1,1,1])
    return f

  def take_action(self, states, actions, step_number=None):
    new_states = self.task.env.take_action(states, actions)
    rewards = [0 for _ in states]
    return new_states, rewards

  def get_optimal_action(self, states, j):
    task = self.task; env = task.env; task_params = self.task_params
    acts = []
    for i in range(task_params.batch_size):
      n, d = env.task.graph.get_action_distance(task.dists[i], states[i])
      a = d == np.min(d)
      acts.append(a)
    acts = np.array(acts)*1
    return acts 

  def get_targets(self, states, j):
    """Returns the optimal action at a location."""
    a = self.get_optimal_action(states, j)
    a = np.expand_dims(a, axis=1)*1
    return {'gt_action': a}

  def get_gamma(self):
    return 0.99 #self.task_params.rl_params.gamma

  def make_vis(self, out_dir, suffix='', prefix=''):
    min_size = 12
    # Make a plot of the episode for environments in this batch.
    fig, _, axes = utils.subplot2(plt, (3, self.task_params.batch_size), (5,5))
    axes = _.T.ravel()[::-1].tolist()
    full_view = self.task.env.task.scaled_views[0]
    vs = self.task.env.task_params.view_scales[0]
    step_size = self.task.env.task_params.step_size
    env = self.task.env

    common_data = self._get_common_data(self.task.init_states)

    for i in range(self.task_params.batch_size):
      ax1 = axes.pop()
      ax1.imshow(np.max(common_data['goal_img_0'][i,...], 2), alpha=0.6, origin='lower')
      ax1.plot((common_data['goal_img_0'][i,...].shape[1]-1)/2., 
        (common_data['goal_img_0'][i,...].shape[0]-1)/2., 'rx')
      
      ax1 = axes.pop()
      ax1.imshow(np.max(common_data['road_0'][i,...], 2), alpha=0.6, origin='lower')
      ax1.plot((common_data['road_0'][i,...].shape[1]-1)/2., 
        (common_data['road_0'][i,...].shape[0]-1)/2., 'rx')
      
      ax2 = axes.pop()
      # Plot 1 with the trajectory on the map.
      ax2.imshow(full_view, alpha=0.6, origin='lower')
      ax = ax2


      all_locs = []
      states = env.episodes[i].states
      
      cmap, sz, lw = 'copper', 40, 0
      loc = env.get_loc_axis(np.array(states).astype(np.int32))[0]
      orien = env.task.graph.nodes[np.array(states).astype(np.int32), 2]
      loc = loc*vs
      ms = ['>', '^', '<', 'v']
      for j in range(4):
        ind = orien == j
        c = np.arange(loc.shape[0])[ind]
        ax.scatter(loc[ind,0], loc[ind,1], c=c, s=sz, cmap=cmap, marker=ms[j],
          edgecolor='k', lw=lw, vmin=0, vmax=loc.shape[0])
      all_locs.append(loc)
     
      for ss, col in zip([self.task.goal_states, self.task.init_states], 'rb'):
        goal_loc = env.get_loc_axis(ss)[0][i:i+1,:]*vs
        goal_orien = env.task.graph.nodes[ss][i,2]
        ax.plot(goal_loc[:,0], goal_loc[:,1], col+ms[goal_orien], ms=10, alpha=0.5) 
        all_locs.append(goal_loc)
      
      all_locs = np.concatenate(all_locs, axis=0)
      min_ = np.min(all_locs, axis=0)
      max_ = np.max(all_locs, axis=0)
      mid_ = (min_+max_)/2.
      sz = np.maximum(1.2*np.max(max_-min_)/2., min_size)
      ax.set_xlim([mid_[0]-sz, mid_[0]+sz])
      ax.set_ylim([mid_[1]-sz, mid_[1]+sz])
      ax.get_xaxis().set_ticks([])
      ax.get_yaxis().set_ticks([])
    out_file_name = os.path.join(out_dir, 
      '{:s}env_vis{:s}.png'.format(prefix, suffix))
    fig.savefig(out_file_name, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close(fig)

def test_cost_map():
  d = factory.get_dataset('campus', 'small')
  name = d.get_imset()[0]
  logging.error(name)
  top_view_param = tle.get_top_view_discrete_env_task_params(
    prob_random=0.2, batch_size=4, map_max_size=200, step_size=1, output_roads=True, 
    ignore_roads=True, top_view=False, fovs=[32])
  e = tle.TopViewDiscreteEnv(dataset=d, name=name, task_params=top_view_param,
    flip=False, rng=np.random.RandomState(0))
  cost_map_task_params = get_cost_map_task_params(batch_size=4, min_dist=10,
    max_dist=40)
  f = CostMap(e, cost_map_task_params)
  rng = np.random.RandomState(0)
  init_states = f.reset(rng)
  f.get_common_data()
  states = init_states
  for i in range(40):
    feats = f.get_features(states)
    logging.error('%s', feats.keys())
    acts = f.get_optimal_action(states, 0)
    gt_actions = f.get_targets(states, 0)
    acts = np.argmax(acts, axis=1)
    states, reward = f.take_action(states, acts)
    logging.error('%s, %s', str(acts), str(states))
  f.make_vis('tmp', '_cost_map_test')

def test_cost_map_sbpd():
  d = factory.get_dataset('sbpd', 'train1')
  name = d.get_imset()[0]
  logging.error(name)
  top_view_param = tle.get_top_view_discrete_env_task_params(
    prob_random=0.1, batch_size=16, fovs=[40, 40, 40], view_scales=[0.0625, 0.25, 0.125], step_size=8,
    output_roads=True, ignore_roads=False, top_view=False)
  
  e = mp_env.MPDiscreteEnv(dataset=d, name=name, task_params=top_view_param,
    flip=False, rng=np.random.RandomState(0))
  
  cost_map_task_params = get_cost_map_task_params(batch_size=16, min_dist=10,
    max_dist=40)
  f = CostMap(e, cost_map_task_params)
  rng = np.random.RandomState(0)
  init_states = f.reset(rng)
  f.get_common_data()
  states = init_states
  for i in range(40):
    feats = f.get_features(states)
    logging.error('%s', feats.keys())
    acts = f.get_optimal_action(states, 0)
    gt_actions = f.get_targets(states, 0)
    acts = np.argmax(acts, axis=1)
    states, reward = f.take_action(states, acts)
    logging.error('%s, %s', str(acts), str(states))
  f.make_vis('tmp', '_sbpd_cost_map_test')
