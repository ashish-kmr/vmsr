import os, sys
import numpy as np
import logging
#import tensorflow as tf
from src import utils
from cfgs import config_common as cc
from env import toy_landmark_env as tle
from env import mp_env
from env import factory
from cfgs import config_manager as cm
from cfgs import config_trajectory_follower as ctf
str_to_float = cm.str_to_float

def get_default_args():
  summary_args = utils.Foo(display_iters=20, test_iters=40,
    arop_full_summary_iters=8, summary_iters=20)
  control_args = utils.Foo(train=False, test=False, reset_rng_seed=False,
    only_eval_when_done=False, test_mode=None, name=None)
  return summary_args, control_args

def get_default_arch_args():
  arch_args = utils.Foo(sample_gt_prob='zero')
  return arch_args

class ConfigManager(ctf.ConfigManager):
  def _mode_hook(self, mode_args, mode_str, args):
    mode_vars = mode_args.process_string(mode_str)
    mode = mode_vars.mode
    k = str_to_float(mode_vars.x_steps[:-1])
    args.arch.num_steps = int(k*args.arch.num_steps)
    assert(mode in ['train', 'val1', 'val2', 'val3', 'bench', 'benche2e']), "mode is not in train, val1, val2." 

    if mode == 'train':
      args.control.train = True
      args.control.sample_action_type = 'sample'
      args.control.mode = 'train'
      # m.sample_action_type
    elif mode == 'val1':
      args.control.test = True
      args.control.mode = 'val'
      args.control.sample_action_type = 'sample'
    elif mode == 'val2':
      args.task.env_task_params.perturb_views=False
      args.task.env_task_params.batch_size=1
      args.task.env_task_params_2.batch_size=1
      args.task.env_task_params_2.add_flips = False
      args.control.test = True
      args.control.mode = 'live'
      args.control.sample_action_type = 'argmax'
      args.arch.sample_gt_prob = 'zero'
      args.arch.anneal_gt_demon = 'zero'
      args.summary.test_iters = 200
    elif mode == 'val3':
      args.control.test = True
      args.control.mode = 'live'
      args.control.sample_action_type = 'argmax'
      args.arch.sample_gt_prob = 'zero'
      args.arch.anneal_gt_demon = 'zero'
      args.task.env_task_params_2.add_flips = False
      args.summary.test_iters = 100
    elif mode == 'bench':
      # forcing batch_size to 1 for bench runs, to ensure that the same paths are generated across all runs.
      args.task.env_task_params.perturb_views=False
      args.task.env_task_params.batch_size=1
      args.task.env_task_params_2.batch_size=1

      args.task.env_task_params_2.add_flips = False
      args.control.test = True
      args.control.mode = 'live'
      args.control.sample_action_type = 'argmax'
      args.control.only_eval_when_done = False 
      args.summary.test_iters = 500
      args.arch.sample_gt_prob = 'zero'
      args.arch.anneal_gt_demon = 'zero'
      args.control.reset_rng_seed = True
    elif mode == 'benche2e':
      args.control.test = True
      args.control.mode = 'live'
      args.control.sample_action_type = 'argmax'
      args.control.only_eval_when_done = True
      args.summary.test_iters = 1000
      args.arch.sample_gt_prob = 'zero'
      args.arch.anneal_gt_demon = 'zero'
      args.control.reset_rng_seed = True
      args.task.env_task_params_2.plan_type = 'custom'
      args.task.env_task_params_2.plan_path = 'output-r2d2/mapper_planner_insitu_v1/bs8_N2en1_20_8_16_18_1_______ms40_goal_es4e2_flip1.v2_rs18_frz1_bn1_dr64_wtadd.dlw1e1_rlw1e0_ent0e0_lr1en4_adam2___clip2/bench_{:s}/plans/00060001/'.format(mode_vars.imset)

    else:
      logging.fatal('Unknown mode: %s.', mode)
      assert(False)
    args.task.dataset = factory.get_dataset(args.task.dataset_name, mode_vars.imset)
    args.task.names = args.task.dataset.get_imset()
    args.control.name = '{:s}'.format(mode_str)
    return args

  def _arch_setup(self):
    args_ = [('ver', 'v0'), ('num_steps', 'ns20'), ('image_cnn', 'rs18'),
    ('freeze_conv', 'frz1'), ('batch_norm', 'bn1'), ('dim_reduce_neurons',
    'dr0'), ('sample_gt_prob', 'zero'), ('map_type', 'map1'), ('dnc', 'dnc2'),
    ('combine_type', 'wtadd'), ('map_source', 'samples'), ('aux_loss', 'aux0'), 
    ('anneal_gt_demon', 'zero'), ('bn_pose', 'bnp1'), ('increment_fn_type', 'sigmoid')] 
    return utils.DefArgs(args_)

  def _arch_hook(self, arch_args, arch_str, args):
    arch_vars = arch_args.process_string(arch_str)
    arch = get_default_arch_args()
    logging.error('arch_vars: %s', arch_vars)
    
    arch.num_steps = int(arch_vars.num_steps[2:])
    arch.image_cnn = arch_vars.image_cnn
    arch.freeze_conv = int(arch_vars.freeze_conv[3:]) > 0
    arch.dim_reduce_neurons = int(arch_vars.dim_reduce_neurons[2:])
    arch.use_bn = int(arch_vars.batch_norm[2:]) > 0
    arch.map_source = arch_vars.map_source
    arch.aux_loss = int(arch_vars.aux_loss[3:])
    arch.anneal_gt_demon = arch_vars.anneal_gt_demon
    arch.bn_pose = int(arch_vars.bn_pose[3:]) > 0
    
    assert(arch_vars.increment_fn_type in ['sigmoid', 'tanh'])
    kk = {'sigmoid': 'sigmoid', 'tanh': 'tanh'}
    arch.increment_fn_type = kk[arch_vars.increment_fn_type]
    assert(arch.anneal_gt_demon in ['zero', 'one'] or arch.anneal_gt_demon[:3] == 'isd')
    # sdemon subsamples demonstration images.
    assert(arch_vars.map_source in ['samples', 'demon'] or arch_vars.map_source[:6] == 'sdemon') 
    
    map_types = {'map1': 'map', 'fsynth': 'fsynth', 'fsynthfc': 'fsynthfc', 'map0': 'none', 'none': 'none'}
    assert(arch_vars.map_type in map_types.keys())
    arch.map_type = map_types[arch_vars.map_type]
      
    arch.combine_type = arch_vars.combine_type
    assert(arch.combine_type in ['gru', 'gtpick', 'max', 'wtadd'])
    
    for x in ['ver', 'sample_gt_prob']:
      setattr(arch, x, getattr(arch_vars, x))
    
    arch.dnc = int(arch_vars.dnc[3:])
    arch.dnc_steps = 10

    if arch.image_cnn == 'rs50':
      arch.image_cnn = 'resnet_v2_50'
    elif arch.image_cnn == 'rs18':
      arch.image_cnn = 'resnet_v2_18'
    elif arch.image_cnn == 'sn':
      arch.image_cnn = 'simple_net'
    elif arch.image_cnn == 'sn5':
      arch.image_cnn = 'simple_net_5'
    elif arch.image_cnn == 'wdw55':
      arch.image_cnn = 'wider_window5_net_5'
    elif arch.image_cnn == 'sn5mp':
      arch.image_cnn = 'simple_net_5_maxp'
    else:
      assert(False), 'image_cnn is not defined.'
    
    args.arch = arch
    return args

  def _task_setup(self):
    args = [('batch_size', 'bs4'), ('noise', 'N1en1'), ('path_length', '20'),
      ('step_size', '128'), ('minD', '4'), ('maxD', '20'), ('num_waypoints', '8'),
      ('base_resolution', 'br1x0'), ('typ', 'sp'), ('history', 'h0'), 
      ('dataset', 'sbpd'), ('perturb_views', 'vp0'), ('modality', 'rgb'), ('mapping_samples', 'ms30'),
      ('dist_type', 'goal'), ('extent_samples', 'es2e2'), ('flips', 'flip0'), ('t_noise','TN0'),('task_typ','forward'),('data_typ','demonstartion'),('replay_buffer','rp0'),('tf_distort','tfd0'),('minQ','mQ20'),('reuseCount','rC4'),('spacious','dilate0'),('multi_act','multi0'),('dilation_cutoff','4'), ('nori','12'), ('hvfov','60')]
    return utils.DefArgs(args)

  def _task_hook(self, task_vars_args, task_str, args):
    task_vars = task_vars_args.process_string(task_str)
    logging.error('task_vars: %s', task_vars)
    
    noise = str_to_float(task_vars.noise[1:])
    t_noise = str_to_float(task_vars.t_noise[2:])
    batch_size = int(task_vars.batch_size[2:])
    step_size = int(task_vars.step_size)
    num_waypoints = int(task_vars.num_waypoints)
    min_dist = int(task_vars.minD)
    max_dist = int(task_vars.maxD)
    mapping_samples = int(task_vars.mapping_samples[2:])
    path_length = int(task_vars.path_length)
    base_resolution = str_to_float(task_vars.base_resolution[2:])
    add_flips = int(task_vars.flips[4:]) > 0
    typ = task_vars.typ
    history = int(task_vars.history[1:])
    perturb_views = int(task_vars.perturb_views[2:]) > 0
    modality = task_vars.modality
    modality = 'disparity' if modality == 'd' else modality
    road_dilate_disk_size = 0
    dist_type = task_vars.dist_type + '_dists'
    extent_samples = int(str_to_float(task_vars.extent_samples[2:]))
    task_typ=task_vars.task_typ
    data_typ=task_vars.data_typ
    if (task_typ=='return'):
      assert(data_typ=='mapping')
    assert(dist_type in ['goal_dists', 'traj_dists'])
    replay_buffer=int(task_vars.replay_buffer[2:])
    tf_distort=int(task_vars.tf_distort[3:])>0
    minQ=int(task_vars.minQ[2:])
    reuseCount=int(task_vars.reuseCount[2:])
    spacious=int(task_vars.spacious[-1])>0
    multi_act=int(task_vars.multi_act[-1])>0
    dilation_cutoff=int(task_vars.dilation_cutoff)
    nori = int(task_vars.nori)
    h_fov = float(task_vars.hvfov)
    v_fov = float(task_vars.hvfov)
    fovs = 32
    vs = 1. / base_resolution
    
    top_view_task_params = tle.get_top_view_discrete_env_task_params(prob_random=noise,
      fovs=[fovs], view_scales=[vs], batch_size=batch_size, ignore_roads=True, 
      output_roads=False, road_dilate_disk_size=road_dilate_disk_size, 
      map_max_size=None, base_resolution=base_resolution, step_size=step_size,
      top_view=True, perturb_views=perturb_views,t_prob_noise=t_noise,replay_buffer=replay_buffer,
      tf_distort=tf_distort,minQ=minQ,reuseCount=reuseCount,spacious=spacious,multi_act=multi_act, dilation_cutoff=dilation_cutoff, nori = nori)
    
    follower_task_params = tle.get_follower_task_params(batch_size=batch_size, 
      history=history, min_dist=min_dist, max_dist=max_dist,
      num_waypoints=num_waypoints, path_length=path_length, add_flips=add_flips,
      typ=typ, data_typ=data_typ, mapping_samples=mapping_samples, share_start=False,
      dist_type=dist_type, extent_samples=extent_samples, plan_type='opt',
      plan_path=None,task_typ=task_typ,replay_buffer=replay_buffer,tf_distort=tf_distort,
      minQ=minQ,reuseCount=reuseCount,spacious=spacious,multi_act=multi_act)
    
    assert(task_vars.dataset in ['campus', 'sbpd','sbpdHD', 'mp3d', 'mp3dHD'])
    if task_vars.dataset == 'campus':
      env_class = tle.TopViewDiscreteEnv
      camera_param = None
      assert(modality in ['rgb'])
      img_channels = 3

    elif task_vars.dataset in ['sbpd','mp3d']:
      env_class = mp_env.MPDiscreteEnv
      img_channels = 2 if modality is 'disparity' else 3
      camera_param = utils.Foo(width=256, height=256, z_near=0.05, z_far=20.0,
        fov_horizontal=h_fov, fov_vertical=v_fov, modalities=[modality],
        img_channels=img_channels, im_resize=224./256.)
    
    elif task_vars.dataset in ['sbpdHD','mp3dHD']:
      task_vars.dataset=task_vars.dataset[0:-2]
      env_class = mp_env.MPDiscreteEnv
      img_channels = 2 if modality is 'disparity' else 3
      camera_param = utils.Foo(width=1024, height=1024, z_near=0.05, z_far=20.0,
        fov_horizontal=h_fov, fov_vertical=v_fov, modalities=[modality],
        img_channels=img_channels, im_resize=1.0)
      #camera_param = utils.Foo(width=1024, height=1024, z_near=0.05, z_far=20.0,
      #  fov_horizontal=60., fov_vertical=60., modalities=[modality],
      #  img_channels=img_channels, im_resize=1.)
      #top_view_task_params.ignore_roads = False
 
    task = utils.Foo()
    task.img_channels = img_channels
    task.seed = 0
    task.dataset_name = task_vars.dataset
    task.env_class = env_class
    task.env_task_params = top_view_task_params
    task.env_class_2 = tle.Follower
    task.env_task_params_2 = follower_task_params
    task.camera_param = camera_param
    task.modality = modality
    task.num_actions = 4
    args.task = task
    args.env_multiplexer = tle.EnvMultiplexer
    return args 
  
  def compile(self):
    names = ['task', 'arch', 'solver', 'mode']
    for n in names:
      self.def_args += [getattr(self, '_'+n+'_setup')()]
      self.name_args += [n]

  def post_hooks(self, args):
    return args

  def pre_hook(self, args):
    args.summary, args.control = get_default_args()
    return args

def test_get_args_for_config():
  # args1 = get_args_for_config('..+train_train1')
  cm_tf = ConfigManager()
  args1 = cm_tf.process_string('..+train_train1')
  args2 = cm_tf.process_string('bs8_N1en1_40_128_4_20_8_br1x0_sp_h0..+train_val')
  #bs4_N1en1_20_128_4_20_8_br1x0_sp_h0_sbpd_vp0_rgb.v0_ns80_128_1_gru_isd100_bn0_dnc0_0_v0_add_act_256_aux0_dimr0_sn.dlw20_rlw1_elw1en0_lr1en3_adam2_MI6e4_SI2e4_noclip_seed0.train_train_1X

  env = args2.env_multiplexer(args2.task, 0, 1)
  rng = np.random.RandomState(19)
  ee = env.sample_env(rng)
  states = ee.reset(rng)
  ee.get_common_data()
  import pdb; pdb.set_trace()

def _test_env():
  """Testing code to make sure that the environment is configured properly."""
  import matplotlib.pyplot as plt
  cm = ConfigManager()
  args = cm.process_string('bs1_N2en1_40_8_16_18_1_____vp0______TN0_forward_demonstartion_____dilate1_multi1.v0_ns40_sn5_frz0_bn1_dr64_one_fsynth_dnc2_gru_demon.dlw1e1_rlw1en1_ent0e0_lr1en4_adam2+train_train1')
  env = args.env_multiplexer(args.task, 0, 1)
  #rng = np.random.RandomState(0)
  rng=np.random.RandomState(19)
  use_info='teacher'
  for ___ in range(1):
    print(___)
    e = env.sample_env(rng)
    init_env_state = e.reset(rng)
    input = e.get_common_data()
    input = e.pre_common_data(input)
    states = []
    states.append(init_env_state)
    targets = []; optimal_actions = []; fs = [];
    for j in range(40):
      f = e.get_features(states[j], j); 
      f = e.pre_features(f);
      fs.append(f)
      #print(fs[-1]['view_xyt'])
      optimal_action = e.get_optimal_action(states[j], j)
      #print(optimal_action)
      target = e.get_targets(states[j], j)
      targets.append(target)
      optimal_actions.append(optimal_action)
      #print(optimal_action)
      #import pdb; pdb.set_trace()
      act_to_take=np.argmax(optimal_action,1)
      if optimal_action[0][3]==1:
        act_to_take=[3]
      new_state, reward = e.take_action(states[j], act_to_take, j)
      #print(states[j][0],new_state[0],np.argmax(optimal_action,1)[0])
      states.append(new_state)
    if ___ >= 0:

      #for __ in range(8):
      for __ in range(1):
        traj=[]
        for i in states:
          traj.append(i[__])
        #import pdb; pdb.set_trace()
        e.task.env.exp_nav.plot(traj,states[-1][__][:],add_str=str(__),only_top=False)
        e.task.env.exp_nav.plot(input[use_info+'_xyt'][__],input[use_info+'_xyt'][__][-1][:],add_str='_coarse_'+str(__),only_top=False)#,plot_img=e.task.env.room_map)
        #e.task.env.exp_nav.plot_view_xyt(traj,dir_name='dense_HD_'+str(___),add_str='student_')
        #e.task.env.exp_nav.plot_view_xyt(input[use_info+'_xyt'][__],dir_name='dense_HD_'+str(___),add_str='teacher_')
        fig, _, axes = utils.subplot2(plt, (8,10), (4,4))
        #print(input['teacher_views'].shape)
        #print(len(axes))
        for i in range(input[use_info+'_views'].shape[1]):
          #print(axes)
          #print(input['teacher_views'].shape)
          ax = axes.pop()
          ax.imshow(input[use_info+'_views'][__,i,:,:,:])
          ax.set_title('{:d}'.format(input[use_info+'_actions'][__,i]))
          ax.set_axis_off()

          ax = axes.pop()
          ax.imshow(fs[i]['view'][__,0,:,:,:,0].astype(np.uint8))
          #print(fs[i]['view_xyt'])
          ax.set_title('Optimal action: {:s}, Target: {:s}'.\
            format(str(optimal_actions[i][__,:]), str(targets[i]['gt_action'][__,0,:])))
          ax.set_axis_off()
        out_file_name = os.path.join('tmp/env_vis/', 'env_vis_{:d}_{:d}.png'.format(___, __))
        fig.savefig(out_file_name, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close(fig)

if __name__ == '__main__':
  _test_env()

