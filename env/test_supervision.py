from _logging import logging
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from env import toy_landmark_env as tle
from render import swiftshader_renderer as sru
from src import utils
from env.mp_env import MPDiscreteEnv, expert_navigator
from env import factory
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
def _test():
  camera_param = utils.Foo(width=256, height=256, z_near=0.05, z_far=20.0,
    fov_horizontal=60., fov_vertical=60., modalities=['rgb'], img_channels=3, 
    im_resize=1.)
  r_obj = sru.get_r_obj(camera_param)

  d = factory.get_dataset('sbpd', 'train1')
  name = d.get_imset()[0]
  logging.error(name)
  
  top_view_param = tle.get_top_view_discrete_env_task_params(
    prob_random=0.0, batch_size=4, view_scales=[0.125], fovs=[64],
    base_resolution=1.0, step_size=8, top_view=True, ignore_roads=False, t_prob_noise=0.2)
  
  e = MPDiscreteEnv(dataset=d, name=name, task_params=top_view_param, 
      flip=False, r_obj=r_obj, rng=np.random.RandomState(0))
  #e = MPDiscreteEnv(dataset=d, name=name, task_params=top_view_param, 
  #  flip=False, r_obj=r_obj, rng=np.random.RandomState(0))
  follower_task_param = tle.get_follower_task_params(
    batch_size=4, min_dist=4, max_dist=20, path_length=40, 
    num_waypoints=8, typ='U')
  f = tle.Follower(e, follower_task_param)
  rng=np.random.RandomState(12)
  actions=[3,3,3,3,2,2,3,3,3,3,3,3,3,3,3,3,1,1,1,0,0]
  goal=e._sample_point_on_map(rng,in_room=True,spacious=True)
  #expert_nav=expert_navigator(e,goal[2],curr_state)
  goal_dist=e.exp_nav._compute_distance_field(goal)
  path=None
  while(path is None):
    curr_state=e.reset(rng,batch_size=1,spacious=True)
    path,actions=e.exp_nav._find_shortest_path(curr_state,goal,goal_dist,80,coarse_map=False,noise_prob=0.0,spacious=True)
  print('curr_state',curr_state)
  print('goal',goal)
  #import pdb; pdb.set_trace()
  _,traj=e.exp_nav._virtual_steps(actions,path[0],goal_dist,traversible=None,noise=0.0, check_collision=False)
  #print(traj)

  print(traj)
  print(actions)
  #for i in range(len(traj)):
  #  e.exp_nav.plot([path[0]]+traj[0:i],goal,goal_dist,actions,plot_img=e.room_map,add_str='actions_noiseless'+str(i))
  
  e.exp_nav.plot([path[0]]+traj,goal,goal_dist,actions,plot_img=e.room_map,add_str='actions_noiseless')
  e.exp_nav.plot(path,goal,goal_dist,actions,plot_img=e.room_map)
def main(_):
  _test() 

if __name__ == '__main__':
  app.run()
