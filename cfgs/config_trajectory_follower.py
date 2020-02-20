import os, sys
import numpy as np
import logging
import tensorflow as tf
from src import utils
from cfgs import config_common as cc
from env import toy_landmark_env as tle
from env import mp_env
from env import factory
from cfgs import config_manager as cm
str_to_float = cm.str_to_float

def get_default_args():
  summary_args = utils.Foo(display_iters=20, test_iters=8,
    arop_full_summary_iters=8, summary_iters=20)
  control_args = utils.Foo(train=False, test=False, reset_rng_seed=False,
    only_eval_when_done=False, test_mode=None, name=None)
  return summary_args, control_args

def get_default_arch_args():
  batch_norm_param = {'center': True, 'scale': True,
    'activation_fn':tf.nn.relu}
  arch_args = utils.Foo(batch_norm_param=batch_norm_param, sample_gt_prob='zero')
  return arch_args

# def process_task_str(task_str):
#   args = [('batch_size', 'bs4'), ('noise', 'N1en1'), ('path_length', '20'),
#     ('step_size', '128'), ('minD', '4'), ('maxD', '20'), ('num_waypoints', '8'),
#     ('base_resolution', 'br1x0'), ('typ', 'sp'), ('history', 'h0')] 
#   task_vars_args = utils.DefArgs(args)
#   task_vars = task_vars_args.process_string(task_str)
#   logging.error('task_vars: %s', task_vars)
# 
#   noise = str_to_float(task_vars.noise[1:])
#   batch_size = int(task_vars.batch_size[2:])
#   step_size = int(task_vars.step_size)
#   num_waypoints = int(task_vars.num_waypoints)
#   min_dist = int(task_vars.minD)
#   max_dist = int(task_vars.maxD)
#   path_length = int(task_vars.path_length)
#   base_resolution = str_to_float(task_vars.base_resolution[2:])
#   add_flips = True
#   typ = task_vars.typ
#   history = int(task_vars.history[1:])
#   road_dilate_disk_size = 0
# 
#   fovs = int(np.round(64))
#   vs = 0.125 / base_resolution
#   
#   top_view_task_params = tle.get_top_view_discrete_env_task_params(prob_random=noise,
#     fovs=[fovs], view_scales=[vs], batch_size=batch_size, ignore_roads=True, 
#     output_roads=False, road_dilate_disk_size=road_dilate_disk_size, 
#     map_max_size=None, base_resolution=base_resolution, step_size=step_size,
#     top_view=True)
#   
#   follower_task_params = tle.get_follower_task_params(batch_size=batch_size, 
#     history=history, min_dist=min_dist, max_dist=max_dist,
#     num_waypoints=num_waypoints, path_length=path_length, add_flips=False,
#     typ=typ)
# 
#   # camera_param = utils.Foo(width=225, height=225, z_near=0.05, z_far=20.0,
#   #   fov_horizontal=60., fov_vertical=60., modalities=['rgb'], img_channels=3)
#   camera_param = None
# 
#   task = utils.Foo()
#   task.seed = 0
#   task.env_task_params = top_view_task_params
#   task.follower_task_params = follower_task_params
#   task.env_class = tle.TopViewDiscreteEnv
#   task.camera_param = camera_param
#   task.num_actions = 4
#   return task
# 
# def process_arch_str(args, arch_str):
#   # This function modifies args.
#   arch = get_default_arch_args()
#   args_ = [('ver', 'v0'), ('num_steps', 'ns80'), ('rnn_units', '128'), ('rnn_n', '1'),
#     ('rnn_type', 'gru'), ('sample_gt_prob', 'isd100'), ('batch_norm', 'bn0'), ('dnc', 'dnc0'),
#     ('dnc_steps', '0'), ('use_vision', 'v0'), ('combine_type', 'add'),
#     ('loss_type', 'act'), ('num_neurons', '256'), ('use_aux_loss', 'aux0')]
#   arch_args = utils.DefArgs(args_)
#   arch_vars = arch_args.process_string(arch_str)
#   logging.error('arch_vars: %s', arch_vars)
#   arch.num_steps = int(arch_vars.num_steps[2:])
#   arch.batch_norm = int(arch_vars.batch_norm[2:]) > 0
#   arch.dnc = int(arch_vars.dnc[3:])
#   arch.dnc_steps = int(arch_vars.dnc_steps)
#   arch.num_neurons = int(arch_vars.num_neurons)
#   arch.use_vision = int(arch_vars.use_vision[1:]) > 0
#   arch.combine_type = arch_vars.combine_type
#   arch.loss_type = arch_vars.loss_type
#   arch.use_aux_loss = int(arch_vars.use_aux_loss[3:])
# 
#   if arch.loss_type == 'qval':
#     arch.use_gt_q_value = True
#   else:
#     arch.use_gt_q_value = False
#   
#   if not arch.use_vision:
#     # Switch off rendering if not needed.
#     args.task.env_task_params.outputs.top_view = False
# 
# 
#   for x in ['rnn_units', 'rnn_n']:
#     setattr(arch, x, int(getattr(arch_vars, x)))
#   for x in ['ver', 'rnn_type', 'sample_gt_prob']:
#     setattr(arch, x, getattr(arch_vars, x))
#   args.arch = arch
#   return args
# 
# def get_args_for_config(config_name):
#   args = utils.Foo()
#   args.summary, args.control = get_default_args()
#   
#   exp_name, mode_str = config_name.split('+')
#   arch_str, solver_str, task_str = exp_name.split('.')
#   logging.error('config_name: %s', config_name)
#   logging.error('arch_str: %s', arch_str)
#   logging.error('task_str: %s', task_str)
#   logging.error('solver_str: %s', solver_str)
#   logging.error('mode_str: %s', mode_str)
# 
#   args.solver = cc.process_solver_str(solver_str)
#   args.task = process_task_str(task_str)
# 
#   args = process_arch_str(args, arch_str)
#   # Train, test, etc.
#   mode_vars = cc.get_mode_vars(mode_str)
#   args = cc.adjust_args_for_mode(args, mode_vars)
#   args.task.dataset = factory.get_dataset('campus', mode_vars.imset)
#   args.task.names = args.task.dataset.get_imset()
# 
#   args.control.name = '{:s}'.format(mode_str)
#   args.env_multiplexer = tle.EnvMultiplexer
# 
#   # Log the arguments
#   logging.error('%s', args)
#   return args

class ConfigManager(cm.ConfigManagerSolver):
  def _mode_setup(self):
    args = [('mode', 'train'), ('imset', 'train'), ('x_steps', '1X')]
    return utils.DefArgs(args)

  def _mode_hook(self, mode_args, mode_str, args):
    mode_vars = mode_args.process_string(mode_str)
    
    mode = mode_vars.mode
    k = str_to_float(mode_vars.x_steps[:-1])
    args.arch.num_steps = int(k*args.arch.num_steps)
    assert(mode in ['train', 'val1', 'val2']), "mode is not in train, val1, val2." 
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
      args.control.test = True
      args.control.mode = 'live'
      args.control.sample_action_type = 'argmax'
      args.arch.sample_gt_prob = 'zero'
    else:
      logging.fatal('Unknown mode: %s.', mode)
      assert(False)
    args.task.dataset = factory.get_dataset(args.task.dataset_name, mode_vars.imset)
    args.task.names = args.task.dataset.get_imset()
    args.control.name = '{:s}'.format(mode_str)
    return args

  def _arch_setup(self):
    args_ = [('ver', 'v0'), ('num_steps', 'ns80'), ('rnn_units', '128'), ('rnn_n', '1'),
      ('rnn_type', 'gru'), ('sample_gt_prob', 'isd100'), ('batch_norm', 'bn0'), ('dnc', 'dnc0'),
      ('dnc_steps', '0'), ('use_vision', 'v0'), ('combine_type', 'add'),
      ('loss_type', 'act'), ('num_neurons', '256'), ('use_aux_loss', 'aux0'), ('dim_reduce_neurons', 'dimr0'),
      ('image_cnn', 'sn'), ('use_teacher_vision', 'tv1')]
    return utils.DefArgs(args_)

  def _arch_hook(self, arch_args, arch_str, args):
    arch_vars = arch_args.process_string(arch_str)
    arch = get_default_arch_args()
    logging.error('arch_vars: %s', arch_vars)
    arch.num_steps = int(arch_vars.num_steps[2:])
    arch.batch_norm = int(arch_vars.batch_norm[2:]) > 0
    arch.dnc = int(arch_vars.dnc[3:])
    arch.dnc_steps = int(arch_vars.dnc_steps)
    arch.num_neurons = int(arch_vars.num_neurons)
    arch.use_vision = int(arch_vars.use_vision[1:]) > 0
    arch.use_teacher_vision = int(arch_vars.use_teacher_vision[2:]) > 0
    arch.dim_reduce_neurons = int(arch_vars.dim_reduce_neurons[4:])
    arch.combine_type = arch_vars.combine_type
    arch.loss_type = arch_vars.loss_type
    arch.use_aux_loss = int(arch_vars.use_aux_loss[3:])
    arch.image_cnn = arch_vars.image_cnn

    if arch.image_cnn == 'rs50':
      arch.image_cnn = 'resnet_v2_50'
      args.task.env_task_params.fovs = [225]
      args.task.camera_param.im_resize = 225./256.
    elif arch.image_cnn == 'rs18':
      arch.image_cnn = 'resnet_v2_18'
      args.task.env_task_params.fovs = [224]
      args.task.camera_param.im_resize = 224./256.
    elif arch.image_cnn == 'sn':
      arch.image_cnn = 'simple_net'
    else:
      assert(False), 'image_cnn is not defined.'

    if arch.loss_type == 'qval':
      arch.use_gt_q_value = True
    else:
      arch.use_gt_q_value = False
    
    if not arch.use_vision:
      # Switch off rendering if not needed.
      args.task.env_task_params.outputs.top_view = False

    for x in ['rnn_units', 'rnn_n']:
      setattr(arch, x, int(getattr(arch_vars, x)))
    for x in ['ver', 'rnn_type', 'sample_gt_prob']:
      setattr(arch, x, getattr(arch_vars, x))
    args.arch = arch
    return args

  def _task_setup(self):
    args = [('batch_size', 'bs4'), ('noise', 'N1en1'), ('path_length', '20'),
      ('step_size', '128'), ('minD', '4'), ('maxD', '20'), ('num_waypoints', '8'),
      ('base_resolution', 'br1x0'), ('typ', 'sp'), ('history', 'h0'), 
      ('dataset', 'campus'), ('perturb_views', 'vp0'), ('modality', 'rgb')] 
    return utils.DefArgs(args)

  def _task_hook(self, task_vars_args, task_str, args):
    task_vars = task_vars_args.process_string(task_str)
    logging.error('task_vars: %s', task_vars)
    
    noise = str_to_float(task_vars.noise[1:])
    batch_size = int(task_vars.batch_size[2:])
    step_size = int(task_vars.step_size)
    num_waypoints = int(task_vars.num_waypoints)
    min_dist = int(task_vars.minD)
    max_dist = int(task_vars.maxD)
    path_length = int(task_vars.path_length)
    base_resolution = str_to_float(task_vars.base_resolution[2:])
    add_flips = True
    typ = task_vars.typ
    history = int(task_vars.history[1:])
    perturb_views = int(task_vars.perturb_views[2:]) > 0
    modality = task_vars.modality
    modality = 'disparity' if modality == 'd' else modality
    road_dilate_disk_size = 0

    fovs = int(np.round(64))
    vs = 0.125 / base_resolution
    
    top_view_task_params = tle.get_top_view_discrete_env_task_params(prob_random=noise,
      fovs=[fovs], view_scales=[vs], batch_size=batch_size, ignore_roads=True, 
      output_roads=False, road_dilate_disk_size=road_dilate_disk_size, 
      map_max_size=None, base_resolution=base_resolution, step_size=step_size,
      top_view=True, perturb_views=perturb_views)
    
    follower_task_params = tle.get_follower_task_params(batch_size=batch_size, 
      history=history, min_dist=min_dist, max_dist=max_dist,
      num_waypoints=num_waypoints, path_length=path_length, add_flips=False,
      typ=typ)
    
    assert(task_vars.dataset in ['campus', 'sbpd'])
    if task_vars.dataset == 'campus':
      env_class = tle.TopViewDiscreteEnv
      camera_param = None
      assert(modality in ['rgb'])
      img_channels = 3

    elif task_vars.dataset == 'sbpd':
      env_class = mp_env.MPDiscreteEnv
      img_channels = 2 if modality is 'disparity' else 3
      camera_param = utils.Foo(width=256, height=256, z_near=0.05, z_far=20.0,
        fov_horizontal=60., fov_vertical=60., modalities=[modality],
        img_channels=img_channels, im_resize=0.25)
      top_view_task_params.ignore_roads = False
 
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
  args2 = cm_tf.process_string('bs8_N1en1_40_8_4_20_8_br1x0_sp_h0.v0_ns40_512_1_gru_zero_bn0_dnc2_20_v1_add_act_1024.dlw1e0_rlw1e0_ent0e0_lr1en4_adam2_MI6e4_SI2x5e4_clip0+train_train1__sbpd')
  import pdb; pdb.set_trace()
