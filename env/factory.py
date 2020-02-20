import os, logging
from env.sbpd import Loader

def get_dataset(name, imset):
  if name == 'sbpd':
    return Loader('stanford_building_parser_dataset', imset)
  elif name == 'mp3d':
    return Loader('mp3d', imset)
  elif name == 'suncg':
    return Loader('suncg', imset)
  else:
    logging.error('Unknown dataset %s', name)

def _test_rooms(ver, imsets):
  for s in imsets:
    d = get_dataset(ver, s)
    ls = d.get_imset()
    for l in ls:
      b = d.load_data(l)
      b._vis_room_dimensions()

def _test_loading(ver, imsets, flip=False):
  for s in imsets:
    d = get_dataset(ver, s)
    ls = d.get_imset()
    for l in ls:
      b = d.load_data(l, flip)
      b._vis_room_dimensions()

def test_mp3d():
  _test_loading('mp3d', ['train6'])

def test_suncg_rooms():
  _test_loading('suncg', ['train48-obj-100'])

def test_suncg_test_rooms():
  _test_loading('suncg', ['test-obj-100'])

def test_suncg_test_rooms_flip():
  _test_loading('suncg', ['test-obj-100'], flip=True)

def test_mp3d_rooms():
  _test_loading('mp3d', ['traincustom'])

def test_sbpd_rooms():
  _test_loading('sbpd', ['all'])

def test_spbd():
  _test_loading('sbpd', ['train1', 'train', 'all'])


def _test_sbpd_env():
  import numpy as np
  from env import toy_landmark_env as tle
  from env.mp_env import MPDiscreteEnv
  dataset = get_dataset('sbpd', 'train1')
  name = dataset.get_imset()[0]
  logging.error(name)
  top_view_param = tle.get_top_view_discrete_env_task_params(
    prob_random=0.1, batch_size=4, map_max_size=200, step_size=8)
  
  e = MPDiscreteEnv(name, dataset, False, task_params=top_view_param)
  
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

def _test_sbpd_env_2():
  from env import toy_landmark_env as tle
  from render import swiftshader_renderer as sru
  from src import utils
  from env.mp_env import MPDiscreteEnv
  import numpy as np
  camera_param = utils.Foo(width=256, height=256, z_near=0.05, z_far=20.0,
    fov_horizontal=60., fov_vertical=60., modalities=['rgb'], img_channels=3, 
    im_resize=1.)
  r_obj = sru.get_r_obj(camera_param)

  d = get_dataset('sbpd', 'train1')
  name = d.get_imset()[0]
  logging.error(name)
  top_view_param = tle.get_top_view_discrete_env_task_params(
    prob_random=0.1, batch_size=4, view_scales=[0.125], fovs=[64],
    base_resolution=1.0, step_size=8, top_view=True, ignore_roads=False)

  e = MPDiscreteEnv(dataset=d, name=name, task_params=top_view_param, 
    flip=False, r_obj=r_obj, rng=np.random.RandomState(0))
  follower_task_param = tle.get_follower_task_params(
    batch_size=4, min_dist=4, max_dist=20, path_length=40, 
    num_waypoints=8, typ='U')
  f = tle.Follower(e, follower_task_param)
