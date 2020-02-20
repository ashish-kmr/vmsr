from __future__ import print_function
import logging
import numpy as np, os, cv2 
import matplotlib.pyplot as plt
from src import utils
import src.file_utils as fu
from src import map_utils as mu 
from env import toy_landmark_env as tle
from tensorflow.python.platform import flags
#from src import graphs
import h5py
import skfmm
import numpy.ma as ma
import copy
from scipy.stats import truncnorm
import re

make_map = mu.make_map
resize_maps = mu.resize_maps
compute_traversibility = mu.compute_traversibility
pick_largest_cc = mu.pick_largest_cc

FLAGS = flags.FLAGS
flags.DEFINE_bool('pre_render', False, """If True then renders all nodes in the
beginning and uses the rendered images, else renders them on the fly. If
pre-rendering can read from file or from memory.""")
flags.DEFINE_bool('use_mp_cache', False, """If pre_render is True then
use_mp_cache being True means that images will be written to a hdf5 file and
read from there. If use_mp_cache is False then images will be kept in memory.
If pre_render is False then this flag is ignored.""")

randperm = False
randsample = False 


def take_freespace_action(angle_value, step_size, states, actions, sim=False):
  """Actions are discrete [0 (stay in place), 1(turn left), 2(turn right),
  3(straight ahead)].
  """
  out_states = []
  batch_size = len(states)
  action_status = []
  for i in range(batch_size):
    action = actions[i]*1
    status = True
    state = states[i]*1
    status = True

    out_state=np.array(list(state))*1.
    du=step_size
    if action == 3:
      angl=out_state[2]
      out_state[0]=(out_state[0]+(np.cos(angl)*du))
      out_state[1]=(out_state[1]+(np.sin(angl)*du))
      out_state[2]=angl
    elif action > 0:
      out_state[2]+=(angle_value[action])
    out_states.append(copy.deepcopy(out_state))
  return out_states

def sim_noise(dx, rng):
  if rng is not None:
    rand_state=rng.randint(1e6)
  else:
    rand_state=None
  if abs(dx)>0:
    rv=truncnorm.rvs(-dx,dx,random_state=rand_state)
  else:
    rv=0.0
  return rv

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

    # class_maps, class_map_names = _get_semantic_maps(
    #   env_paths['class_map_folder'], name, map, flip)
    # self.class_maps = class_maps
    # self.class_map_names = class_map_names

    self.env_paths = env_paths 
    self.shapess = shapess
    self.map = map
    self.traversible = map.traversible*1
    self.name = name 
    self.full_name = dataset.ver + '-' + name + '-' + 'flip{:d}'.format(flip)
    self.flipped = flip
    self.renderer_entitiy_ids = []
    if self.restrict_to_largest_cc:
      self.traversible = pick_largest_cc(self.traversible)
   
    
    free_xy=np.array(np.nonzero(self.traversible)).transpose()
    room_dims = self._get_room_dimensions(env_paths['room_dimension_file'], 
      env.resolution, map.origin, flip=flip)
    room_regex='^((?!hallway).)*$'
    room_dims = self._filter_rooms(room_dims, room_regex)
    self.room_dims = room_dims
    room_list = self._label_nodes_with_room_id(free_xy,room_dims)
    room_idx = (room_list>-1)
    self.free_room_xy = free_xy[room_idx[:,0]][:]
    if (self.free_room_xy.shape[0]==0):
      print('\n\n\nENV IS BAD',name,'\n\n\n')
    
    self.room_xy = copy.deepcopy(self.traversible)
    
    #blocked_xy=np.array(np.nonzero(~self.traversible)).transpose()
    x_l,y_l=np.meshgrid(range(self.room_xy.shape[0]),range(self.room_xy.shape[1]))
    x_l=x_l.reshape([-1,1])
    y_l=y_l.reshape([-1,1])
    blocked_xy=np.concatenate([x_l,y_l],axis=1)
    room_list_on_blocked=self._label_nodes_with_room_id(blocked_xy.astype(np.float32),room_dims,width=10)
    room_idx_blocked=(room_list_on_blocked>-1)
    idx_lst=blocked_xy[room_idx_blocked[:,0]]
    self.room_xy[idx_lst[:,0],idx_lst[:,1]]=True
    #import pdb; pdb.set_trace()

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

  def _get_room_dimensions(self,file_name, resolution, origin, flip=False):
    if fu.exists(file_name):
      a = utils.load_variables(file_name)['room_dimension']
      names = [names_val for names_val in a.keys()]
      dims = np.concatenate([a_val for a_val in a.values()], axis=0).reshape((-1,6))
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
      out = None
    return out 

  def get_salt_string(self):
    str_ = 'robot-{:s}/env-{:s}/{:s}-flip{:d}-ms{:03d}'.format(
      utils.get_hash(self.robot), utils.get_hash(self.env), self.name, 
      self.flipped, int(np.round(1000*self.materials_scale))) 
    return str_ 

  def set_r_obj(self, r_obj):
    self.r_obj = r_obj

  def load_building_into_scene(self):
    assert(self.shapess is not None)
    
    # Loads the scene.
    self.renderer_entitiy_ids += self.r_obj.load_shapes(self.shapess, 
      'flipped{:d}'.format(self.flipped))
    # Free up memory, we dont need the mesh or the materials anymore.
    self.shapess = None
  
  def add_shape_at_location(self, loc_xy, shape):
    name_suffix = '_{:0.6f}{:0.6f}{:0.6f}'.format(loc_xy[0], loc_xy[1], loc_xy[2])
    entity_ids = self.r_obj.load_shapes([shape], name_suffix=[name_suffix], trans=[loc_xy])
    self.renderer_entitiy_ids += entity_ids
    return entity_ids

  def del_entities(self, entity_ids):
    self.r_obj.del_shapes(entity_ids)
    for e in entity_ids:
      id = self.renderer_entitiy_ids.index(e)
      self.renderer_entitiy_ids.pop(id)

  # def add_entity_at_nodes(self, nodes, height, shape):
  #   xyt = self.to_actual_xyt_vec(nodes)
  #   nxy = xyt[:,:2]*1.
  #   nxy = nxy * self.map.resolution
  #   nxy = nxy + self.map.origin
  #   Ts = np.concatenate((nxy, nxy[:,:1]), axis=1)
  #   Ts[:,2] = height; Ts = Ts / 100.;

  #   # Merge all the shapes into a single shape and add that shape.
  #   shape.replicate_shape(Ts)
  #   entity_ids = self.r_obj.load_shapes([shape])
  #   self.renderer_entitiy_ids += entity_ids
  #   return entity_ids

  # def add_shapes(self, shapes):
  #   scene = self.r_obj.viz.scene()
  #   for shape in shapes:
  #     scene.AddShape(shape)

  # def add_materials(self, materials):
  #   scene = self.r_obj.viz.scene()
  #   for material in materials:
  #     scene.AddOrUpdateMaterial(material)
  
  def to_actual_xyt(self, pqr):
    """Converts from node array to location array on the map."""
    #print(pqr)
    out = pqr*1.
    # p = pqr[:,0:1]; q = pqr[:,1:2]; r = pqr[:,2:3];
    # out = np.concatenate((p + self.map.origin[0], q + self.map.origin[1], r), 1) 
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
      #print(xyt)
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


class expert_navigator():
 
  def __init__(self,env):
    self.env=env
    self.map={}
    self.map['traversible_cc']=self.env.task.road
    self.map['coarse_map']=self.env.room_map
    self.map['resolution']=5 #attention
    self.du=self.env.task.step_size #in cm attention
    self.noise=0 # attention
    self.n_ori=self.env.n_ori
    self.angle_value=self.env.angle_value
    self.dt=np.pi/self.n_ori
    num_rots = int(self.n_ori/2)
    self.action_list=self.search_actions(num_rots)
    self.obst_dist_fmm=self._compute_obst_distance_field()

  def _compute_obst_distance_field(self, t=None):
    if t is None:
      t = self.map['traversible_cc'] 
    #print('masked',t.shape)
    masked_t = ma.masked_values(t*1, 0) 
    idx_lst=np.argwhere(np.invert(t))
    masked_t[idx_lst[:,0],idx_lst[:,1]]=0
    #import pdb; pdb.set_trace()
    # scaling as per the reward scaling. 
    dd = skfmm.distance(masked_t, dx=1)#dx=self.map['resolution']/self.du) #self.map['resolution'])  
    dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan))) 
    dd = ma.filled(dd, np.max(dd)+1) 
    return dd

  def _compute_distance_field(self, goal, t=None):
    if t is None:
      t = self.map['traversible_cc'] 
    #print('masked',t.shape)
    masked_t = ma.masked_values(t*1, 0) 
    #print('goal',goal,masked_t.shape,goal)
    goal_x, goal_y = int(goal[0]),int(goal[1])#_map_to_point((goal[0]), (goal[1])) # point on map attention 
    masked_t[goal_x, goal_y] = 0 
    # scaling as per the reward scaling. 
    dd = skfmm.distance(masked_t, dx=1)#dx=self.map['resolution']/self.du) #self.map['resolution'])  
    dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan))) 
    dd = ma.filled(dd, np.max(dd)+1) 
    return dd

  def _in_obs(self,x,y):
    return not self.map['traversible_cc'][int(x),int(y)]
 
  def get_frames(self,given_states, density = 1, lim=None):
    states = []
    init_state = given_states[0]
    for st in given_states[1:]:
        for itr in range(density):
          alpha=float(itr)/density
          mid_st=list((1.0-alpha)*np.array(init_state)+alpha*np.array(st))
          states.append(mid_st)
        init_state = st

    if lim is not None and states.shape[0]<lim:
      repeat_state=states[-1]
      states=np.concatenate([states,np.repeat([repeat_state],lim-states.shape[0],axis=0)],axis=0)
    rendered_img=self.env.render_views(states)[0]
    rendered_img = rendered_img[:,:,:,::-1] 
    return rendered_img

  def _virtual_steps(self, u_list,state,goal_dist,traversible=None,noise=0.0, check_collision=True, rng=None):

    """Actions are discrete [0 (stay in place), 1(turn left), 2(turn right),
    3(straight ahead)].
    """
    if traversible is None:
      traversible=self.map['traversible_cc']
    #print(self.task_params.step_size)
    angle_value=self.env.angle_value
    batch_size = len(u_list)
    action_status = []
    boundary_limits=traversible.shape
    #print(boundary_limits)
    x,y,t=state
    out_states = []
    cost_start = goal_dist[int(x), int(y)] #Actual distance in cm.
    collision_reward=0
    du=self.env.task.step_size
    for i in range(batch_size):
      action = u_list[i]
      x_new, y_new, t_new = x*1., y*1. , t*1.
      du_noise=sim_noise(noise*float(du),rng)
      dt_noise=sim_noise(noise*float(2.0*np.pi/self.n_ori),rng)
      if action == 3:
        angl=t
        du+=du_noise
        angl+=dt_noise
        x_new=(x+(np.cos(angl)*du))
        y_new=(y+(np.sin(angl)*du))
        t_new=angl
      elif action > 0:
        t_new=t+angle_value[action]+dt_noise
      
      if (np.array([int(x_new),int(y_new)])<np.array(boundary_limits)).all() and \
              (np.array([int(x_new),int(y_new)])>=np.array([0,0])).all() and \
              (traversible[int(x_new),int(y_new)]==True or not check_collision) :
        x,y,t=x_new,y_new,t_new
        out_states.append(([x,y,t]))
      else:
        collision_reward=-1
        out_states.append([x,y,t])
        break

    cost_end = goal_dist[int(x), int(y)]
    if(self.compare_goal([x,y,t],goal_dist)):
      reward_near_goal = 1.
    else:
      reward_near_goal=0
    costs = (cost_end - cost_start)

    reward = -costs + reward_near_goal + collision_reward
    return reward, (out_states)

  def compare_goal(self,a,goal_dist):
    x,y,t=a
    cost_end = goal_dist[int(x), int(y)]
    dist = cost_end*1.
    reward_near_goal = 0.
    if dist < self.du*1:
      return True
    return False

  def _find_shortest_path(self,init,goal,goal_dist_inp,limit=1000,coarse_map=False, noise_prob=0.0,rng=None,spacious=False):
    if coarse_map:
      goal_dist=self._compute_distance_field(goal,self.map['coarse_map'])
    else:
      goal_dist=goal_dist_inp

    path=[init[0]]
    actions=[]
    curr_step=0
    curr_state=list(init[0])
    exit_ = False
    #print('curr_state',curr_state)
    while(not self.compare_goal(curr_state,goal_dist)):
      curr_step+=1
      action_set,state_set=self.find_best_action_set(curr_state,goal_dist,self.map['coarse_map'],noise_prob,rng=rng,spacious=spacious)
      curr_state=list(state_set[-1])
      #slow op
      path+=state_set
      actions+=action_set
      #slow op
      if (len(path)>1000):
          break
 
    if limit==1000:
      return path,actions
    elif self.compare_goal(curr_state, goal_dist) and len(path) > limit:
        return path[-limit:], actions[-(limit-1):]
    else:
      return None, None
    
    #print(limit,self.compare_goal(curr_state))
    return self.goal_dist, path[0:limit],actions[0:limit-1] 


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

  def get_obst_dist(self,state):
    return self.obst_dist_fmm[int(state[0]),int(state[1])]

  def find_best_action_set(self,state,goal_dist,traversible=None,noise_prob=0.0,rng=None,spacious=False,\
          multi_act=False):
    action_list=self.action_list
    best_list=[0]
    max_margin=0
    obst_dist=[]
    best_reward,state_list=self._virtual_steps(best_list,state,goal_dist,traversible,noise_prob,rng=rng)
    best_reward=0
    max_margin_state=state_list
    max_margin_act=[0]
    feasible_acts=[]
    feasible_states=[]
    #print(best_reward,)
    for a_list in action_list:
      rew,st_lst = self._virtual_steps(a_list,state,goal_dist,traversible,noise_prob,rng=rng)
      #print(rew,)
      if rew > best_reward:
        best_list=a_list
        best_reward=rew
        state_list=(st_lst)
      if rew>self.env.dilation_cutoff:
        current_margin=self.get_obst_dist(st_lst[-1])
        if current_margin>max_margin:
            max_margin=current_margin
            max_margin_state=st_lst
            max_margin_act=a_list
      if rew>0:
        feasible_acts.append(a_list)
        feasible_states.append(st_lst)
    #print('best_rew',best_reward)
    
    #np.set_printoptions(threshold='nan')
    #np.set_printoptions(linewidth=280)
    #np.set_printoptions(precision=1)
    #print(self.goal_dist[int(x)-10:int(x)+10,int(y)-10:int(y)+10])
    #best_list.append(0)

    if not (len(best_list)==len(state_list)):
        print(len(best_list),len(state_list))
    if multi_act:
        return feasible_acts, feasible_states
    else:
        if not spacious or (len(max_margin_act)==1 and max_margin_act[0]==0):
            return best_list,state_list
        else:
            return max_margin_act,max_margin_state

  def lim_test(self, x, sz, limx): 
      if (x-sz < 0):
          xlow = 0
      elif (x + sz >= limx):
          xlow = limx - 2*sz
      else:
          xlow = x - sz
      return xlow

  def plot(self,traj,goal,goal_dist=None,actions=None,only_top=False,add_str='',plot_img=None, save_img = True, crop = None):
    
    if plot_img is None:
      plot_img=self.map['traversible_cc']
    plot_img.astype(int)
    plot_img=plot_img*255
    dil_lim=2
    #print(goal)
    for i in range(len(traj)):
      st=traj[i]
      for x1 in range(int(st[0])-dil_lim,int(st[0])+dil_lim):
        for x2 in range(int(st[1])-dil_lim,int(st[1])+dil_lim):
          plot_img[min(max(0,x1),plot_img.shape[0]),min(max(0,x2),plot_img.shape[1])]=190
    
    dil_lim=5
    st=goal
    for x1 in range(int(st[0])-dil_lim,int(st[0])+dil_lim):
      for x2 in range(int(st[1])-dil_lim,int(st[1])+dil_lim):
        plot_img[x1,x2]=50
    #for i in self.traj:
    #  plot_img[]
    if save_img:
        if goal_dist is not None:
          cv2.imwrite('output/debug/fmm_distance_'+add_str+'.png',goal_dist)
        cv2.imwrite('output/debug/output_top_view_'+add_str+'.png',plot_img)
    else:
        xmax, ymax = plot_img.shape
        xlow = self.lim_test(int(np.mean(traj,0)[0]), 100, xmax)
        ylow = self.lim_test(int(np.mean(traj,0)[1]), 100, ymax)
        plot_img = plot_img[xlow:xlow + 200, ylow:ylow + 200]
        return plot_img
    np.set_printoptions(threshold='nan')
    np.set_printoptions(linewidth=280)
    np.set_printoptions(precision=1)
    if not only_top:
      img=self.env.render_views(traj)[0]
      count=0
      if actions is not None:
          actions.append(0)
      #print(len(img),len(traj))
      for im in img:
          act_str=''
          if actions is not None:
              act_str=str(actions[count])
          cv2.imwrite('output/debug/test_nograph_'+str(count)+'_'+act_str+'.jpg',im[:,:,::-1])
          count+=1
      if actions is not None:
        actions=actions[:-1]

    #print(self.goal_dist[self.goal[0]-40:self.goal[0]+20,self.goal[1]-20:self.goal[1]+20])
    
  def plot_slam(self,traj,path,density,fname='teacher',write_xyt=True,count=0):
    init_state=traj[0]
    if write_xyt:
      f=open(path+'/../'+fname+'_xyt.txt','w')
    if len(traj)==1:
      mid_st=init_state
      im=self.env.render_views([mid_st])
      cv2.imwrite(path+'/'+fname+'_{:04d}'.format(count)+'.jpg',im[0][0][:,:,::-1])
    else:
      for st in traj[1:]:
        for itr in range(density):
          alpha=float(itr)/density
          mid_st=list((1.0-alpha)*np.array(init_state)+alpha*np.array(st))
          im=self.env.render_views([mid_st])
          cv2.imwrite(path+'/'+fname+'_{:04d}'.format(count)+'.jpg',im[0][0][:,:,::-1])
          count+=1
          l1=[str(itr_st) for itr_st in mid_st]
          if write_xyt:
              f.write(','.join(l1)+'\n')
        init_state=st
      if write_xyt:
        f.close()


class MPDiscreteEnv(tle.DiscreteEnv):
  """Observation is the first person view of the environment.
  Actions are simple grid world actions.
    - Rotate left, right, move straight stay in place.
    - With some probability it stays in place.
  """


  def __init__(self, name, dataset, flip, task_params, rng=None, r_obj=None):
    self.dilation_cutoff=task_params.dilation_cutoff
    self.rng=rng
    self.task = utils.Foo()
    self.task_params = task_params
    print(self.task_params)
    self.task.building = dataset.load_data(name, flip=flip)
    self.task.road = self.task.building.traversible
    self.r_obj = r_obj
    self.resolution=5
    self.n_ori=self.task_params.nori
    self.angle_value=[0,2.0*np.pi/self.n_ori,-2.0*np.pi/self.n_ori,0]
    self.task.step_size=self.task_params.step_size
    print(task_params)
    #self._compute_graph()
    self._setup_noise()
    self._preprocess_for_task()
    self.free_xy=None
    self.free_rooms=self.task.building.free_room_xy
    self.free_xy_spacious=None


    #import pdb; pdb.set_trace()
    self.room_map=self.task.building.traversible#self.task.building.room_xy
    #print(self.free_rooms.shape)
    #self.task.graph = graphs.OrientedGridGraph(self.task.road,step_size=self.task.step_size) 
    self.exp_nav=expert_navigator(self)

  #attention to_actual_xyz
  def _map_to_point(self, x, y):
    r = self.resolution
    o = [0,0]#attention self.task_params['origin']
    x, y = x*r, y*r
    x, y = x + o[0], y + o[1]
    return x, y

  def _sample_point_on_map(self, rng, free_xy=None, in_room=False, spacious=False):
    if free_xy is None and self.free_xy is None and not in_room:
      self.free_xy=np.array(np.nonzero(self.task.road)).transpose()
    if free_xy is None and not in_room:
      free_xy = self.free_xy
    if free_xy is None and in_room:
      free_xy=self.free_rooms
    if spacious: 
      if self.free_xy_spacious is None:
        obst_margins=self.exp_nav.obst_dist_fmm[free_xy[:,0],free_xy[:,1]]  
        idx_i=np.where(obst_margins>5)[0]
        free_xy=free_xy[idx_i,:]
        self.free_xy_spacious=np.copy(free_xy)
      else:
        free_xy=self.free_xy_spacious
    
    #import pdb; pdb.set_trace()
    #print('road shape',self.task.road.shape)
    start_id = rng.choice(free_xy.shape[0])
    start_raw = free_xy[start_id]#*1. <-- attention
    #print(start_raw)
    start = start_raw #+ (rng.rand(2,)-0.5) <-- attention
    #start[0], start[1] = self._map_to_point(*start)
    #print(start)
    assert(self.task.road[start[0], start[1]])
    return list(start)+[0]

  def _preprocess_for_task(self):
    # Pre-Render all the nodes in this building for faster training.
    # self.task.rendered_imgs = np.zeros((self.task.graph.nodes.shape[0], 64, 64, 3), dtype=np.float32)
    
    building = self.task.building
    #if FLAGS.pre_render:
    if False:
      self.task.rendered_imgs_handle = None
      self.task.rendered_imgs = None
      im_size = int(np.round(self.r_obj.im_resize * self.r_obj.height))
    
      
      if FLAGS.use_mp_cache:
        render_salt = self.r_obj.get_salt_string()
        env_salt = self.task.building.get_salt_string()
        node_salt = self.task.graph.get_salt_string()
        # CACHE_BASE_DIR = '/home/eecs/sgupta/landmarks-cache/'
        CACHE_BASE_DIR = 'cache'
        cache_file = os.path.join(CACHE_BASE_DIR, render_salt, node_salt, env_salt) + '.h5'
        dir_name = os.path.dirname(cache_file)
        utils.mkdir_if_missing(dir_name)
        # cache_file = os.path.join('cache', 
        #   '{:s}_{:d}_{:d}_{:s}.h5'.format(building.name, 
        #     self.task.graph.nodes.shape[0], im_size, self.r_obj.modality))
        render = not os.path.exists(cache_file)
      else:
        render = True

      if render and self.r_obj is not None:
        building.set_r_obj(self.r_obj)
        building.load_building_into_scene()
        logging.error('Building: %s, Num nodes: %d', 
          self.task.building.name, self.task.graph.nodes.shape[0])
        logging.error('Rendering all nodes.')
        imgs = building.render_nodes(self.task.graph.nodes)
        self.r_obj.clear_scene()
        if FLAGS.use_mp_cache:
          # Write down things in a hdf5 file.
          with h5py.File(cache_file, 'w') as f:
            for i, img in enumerate(imgs):
              f.create_dataset('img{:06d}'.format(i), data=img)
          logging.error('Wrote images to cache file: %s', cache_file)

      if FLAGS.use_mp_cache:
        logging.error('Loading images from cache at %s.', cache_file)
        self.task.rendered_imgs_handle = h5py.File(cache_file, 'r')
        if randperm:
          logging.error('mp_env: using randperm')
          assert(not randsample)
          rng_ = np.random.RandomState(0)
          self.task.shuffle_ind = rng_.permutation(self.task.graph.nodes.shape[0])
        elif randsample:
          logging.error('mp_env: using randsample')
          self.task.randsample_rng = np.random.RandomState(0)
        imgs = []
      else:
        self.task.rendered_imgs = np.array(imgs)
    else: 
      building.set_r_obj(self.r_obj)
      building.load_building_into_scene()
      #logging.error('Loading Building: %s, Num nodes: %d into scene', 
      #  self.task.building.name, self.task.graph.nodes.shape[0])
    # Release mesh.
    if building.shapess is not None:
      building.shapess = None
    
    # Prepare scaled_views and scaled_roads for visualizations.
    view = (self.task.road*255).astype(np.uint8)
    view = np.expand_dims(view, 2)*np.ones((1,1,3), dtype=np.uint8)
    self.task.view = view
    self.task.scaled_views = resize_maps(self.task.view,
      self.task_params.view_scales, 'antialiasing')
    self.task.scaled_roads = resize_maps((self.task.road*255).astype(np.uint8), 
      self.task_params.view_scales, 'antialiasing')
    # print(np.max(np.array(imgs)))
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # for i in range(self.task.rendered_imgs.shape[0]): 
    #   file_name = os.path.join('tmp/area1_vis/{:06d}.jpg'.format(i)); 
    #   cv2.imwrite(file_name, self.task.rendered_imgs[i,...].astype(np.uint8))
  
  def render_views(self, states, perturbs = None):
    # Renders out the view from the state.
    states_flipped=[]
    for st in states:
      #print(st)
      states_flipped.append(np.array([st[1],st[0],-st[2]+np.pi/2]))

    imgs = self.task.building.render_views(states_flipped, perturbs = perturbs)
    views = [np.array(imgs)]
    return views

  def take_freespace_action(self, states, actions, sim=False):
    """Actions are discrete [0 (stay in place), 1(turn left), 2(turn right),
    3(straight ahead)].
    """
    out_states = []
    batch_size = len(states)
    action_status = []
    angle_value=self.angle_value
    for i in range(batch_size):
      action = actions[i]*1
      status = True
      state = states[i]*1
      status = True

      out_state=np.array(list(state))*1.
      du=self.task.step_size
      if action == 3:
        angl=out_state[2]
        out_state[0]=(out_state[0]+(np.cos(angl)*du))
        out_state[1]=(out_state[1]+(np.sin(angl)*du))
        out_state[2]=angl
      elif action > 0:
        out_state[2]+=(angle_value[action])
      out_states.append(copy.deepcopy(out_state))
    return out_states
  
  def take_action_pos(self, state, rot, step):
    """Actions are discrete [0 (stay in place), 1(turn left), 2(turn right),
    3(straight ahead)].
    """
    boundary_limits=self.task.road.shape
    out_state=np.array(list(state))*1.
    du=step
    angl=out_state[2]
    angl+=rot
    #print(angl,np.cos(angl),np.sin(angl))
    out_state[0]=(out_state[0]+(np.cos(angl)*du))
    out_state[1]=(out_state[1]+(np.sin(angl)*du))
    out_state[2]=angl
    if (np.array(out_state[0:2])<np.array(boundary_limits)).all() and \
            (np.array(out_state[0:2])>=np.array([0,0])).all() and \
            self.task.road[int(out_state[0]),int(out_state[1])]==True  :
      pass
    else:
      out_state=np.array(list(state))
    return out_state

  def take_action(self, states, actions, sim=False):
    """Actions are discrete [0 (stay in place), 1(turn left), 2(turn right),
    3(straight ahead)].
    """
    #print(self.task_params.step_size)
    #print('states',states)
    out_states = []
    prob_random = self.task_params.noise_model.prob_random
    episodes = self.episodes
    batch_size = len(states)
    prob_random = self.task_params.noise_model.prob_random
    action_status = []
    boundary_limits=self.task.road.shape
    angle_value=self.angle_value
    #print(boundary_limits)
    for i in range(batch_size):
      action = actions[i]*1
      status = True
      state = states[i]*1
      rng = episodes[i].rng_noise
      status = True

      out_state=np.array(list(state))*1.
      du=self.task.step_size
      du_noise=sim_noise(prob_random*float(du),rng)
      dt_noise=sim_noise(prob_random*float(2.0*np.pi/self.n_ori),rng)
      if action == 3:
        angl=out_state[2]
        du+=du_noise
        angl+=dt_noise
        #print(angl,np.cos(angl),np.sin(angl))
        out_state[0]=(out_state[0]+(np.cos(angl)*du))
        out_state[1]=(out_state[1]+(np.sin(angl)*du))
        out_state[2]=angl
      elif action > 0:
        out_state[2]+=(angle_value[action]+dt_noise)
      if (np.array(out_state[0:2])<np.array(boundary_limits)).all() and (np.array(out_state[0:2])>=np.array([0,0])).all() and self.task.road[int(out_state[0]),int(out_state[1])]==True  :
        pass
      else:
        out_state=np.array(list(state))
      out_states.append(copy.deepcopy(out_state))
      if not sim:
        episodes[i].states.append(out_state*1.)
        episodes[i].executed_actions.append(actions[i]*1.)
        episodes[i].action_status.append(status)
    #print('in func',angle_value[actions[0]],actions[0],states[0],out_states[0])
    return out_states


  def reset(self, rng, init_states=None, batch_size=None,spacious=False):
    if batch_size is None:
      batch_size = self.task_params.batch_size
    assert(init_states is None or batch_size == len(init_states))

    episodes = []
    out_init_states = []
    for i in range(batch_size):
      rng_i = np.random.RandomState(rng.randint(1e6))
      rng_noise = np.random.RandomState(rng.randint(1e6))
      
      if init_states is None:
        #init_state=np.array([386,240,-np.pi/2])      
        init_state=self._sample_point_on_map(rng,spacious=spacious)
        #print('init',init_state)
      else:
        init_state = init_states[i]
      
      episode = utils.Foo(rng=rng_i, rng_noise=rng_noise, states=[init_state],
        executed_actions=[], action_status=[])
      episodes.append(episode)
      out_init_states.append(init_state)

    self.episodes = episodes
    # State for the agent is the 2D location on the map, (x,y,theta). 
    return out_init_states



