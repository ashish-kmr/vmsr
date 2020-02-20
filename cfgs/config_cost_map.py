import os, sys
import numpy as np
import logging
import tensorflow as tf
from src import utils
from cfgs import config_common as cc
from env import cost_map_env as cme 
from env import toy_landmark_env as tle
from env import mp_env
from env import factory
from cfgs import config_manager as cm
from cfgs import config_trajectory_follower as ctf
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

class ConfigManager(ctf.ConfigManager):
  
  def _arch_setup(self):
    args_ = [('ver', 'v0'), ('num_steps', 'ns80'), ('rnn_units', '128'), ('rnn_n', '1'),
      ('rnn_type', 'gru'), ('sample_gt_prob', 'isd100'), ('batch_norm', 'bn0'),
      ('dnc', 'dnc0'), ('dnc_steps', '0'), ('use_gumbel', 'gmbl0'), ('gumbel_temp', '0'),
      ('gumbel_hard', 'gh0'), ('vin', 'vin0')]
    return utils.DefArgs(args_)

  def _arch_hook(self, arch_args, arch_str, args):
    arch_vars = arch_args.process_string(arch_str)
    arch = get_default_arch_args()
    logging.error('arch_vars: %s', arch_vars)
    arch.num_steps = int(arch_vars.num_steps[2:])
    arch.batch_norm = int(arch_vars.batch_norm[2:]) > 0
    arch.dnc = int(arch_vars.dnc[3:])
    arch.dnc_steps = int(arch_vars.dnc_steps)
    arch.actual_vin = int(arch_vars.vin[3:]) > 0

    arch.fr_neurons, arch.fr_inside_neurons, arch.fr_stride = 32, 32, [1, 1, 1, 2]
    arch.vin_num_iters, arch.vin_val_neurons, arch.vin_action_neurons, \
    arch.vin_ks, arch.vin_share_wts = 20, 1, 4, 3, True
    # arch.vin_num_iters, arch.vin_val_neurons, arch.vin_action_neurons, \
    # arch.vin_ks, arch.vin_share_wts = 20, 8, 8, 3, False
    arch.use_gumbel = int(arch_vars.use_gumbel[4:]) > 0
    arch.gumbel_temp = str_to_float(arch_vars.gumbel_temp)
    arch.gumbel_hard= int(arch_vars.gumbel_hard[2:]) > 0
    

    for x in ['rnn_units', 'rnn_n']:
      setattr(arch, x, int(getattr(arch_vars, x)))
    for x in ['ver', 'rnn_type', 'sample_gt_prob']:
      setattr(arch, x, getattr(arch_vars, x))
    args.arch = arch
    return args

  def _task_setup(self):
    args = [('batch_size', 'bs4'), ('noise', 'N1en1'), ('step_size', '128'),
      ('minD', '4'), ('maxD', '20'), ('base_resolution', 'br1x0'), ('typ',
      'rng'), ('dataset', 'sbpd'), ('num_ref_points', 'r0')]
    return utils.DefArgs(args)

  def _task_hook(self, task_vars_args, task_str, args):
    task_vars = task_vars_args.process_string(task_str)
    logging.error('task_vars: %s', task_vars)
    
    noise = str_to_float(task_vars.noise[1:])
    batch_size = int(task_vars.batch_size[2:])
    step_size = int(task_vars.step_size)
    min_dist = int(task_vars.minD)
    max_dist = int(task_vars.maxD)
    num_ref_points = int(task_vars.num_ref_points[1:])
    base_resolution = str_to_float(task_vars.base_resolution[2:])
    typ = task_vars.typ
    road_dilate_disk_size = 0

    fovs = [65]
    vs = [0.25 / base_resolution]
    
    top_view_task_params = tle.get_top_view_discrete_env_task_params(
      prob_random=noise, fovs=fovs, view_scales=vs, batch_size=batch_size,
      ignore_roads=True, output_roads=True,
      road_dilate_disk_size=road_dilate_disk_size, map_max_size=None,
      base_resolution=base_resolution, step_size=step_size, top_view=False,
      perturb_views=False)
    
    cost_map_task_params = cme.get_cost_map_task_params(batch_size=batch_size, 
      history=0, min_dist=min_dist, max_dist=max_dist, add_flips=False, typ=typ, 
      num_ref_points=num_ref_points)
    
    assert(task_vars.dataset in ['campus', 'sbpd'])
    if task_vars.dataset == 'campus':
      env_class = tle.TopViewDiscreteEnv
      camera_param = None

    elif task_vars.dataset == 'sbpd':
      env_class = mp_env.MPDiscreteEnv
      camera_param = None;
      top_view_task_params.ignore_roads = False
 
    task = utils.Foo()
    task.camera_param = camera_param
    task.seed = 0
    task.dataset_name = task_vars.dataset
    task.num_actions = 4
    
    task.env_class = env_class
    task.env_task_params = top_view_task_params
    task.env_class_2 = cme.CostMap
    task.env_task_params_2 = cost_map_task_params 
    
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
  args2 = cm_tf.process_string('bs4_N1en1_128_4_20_br1x0_rng_sbpd.v0_ns80_128_1_gru_isd100_bn0_dnc0_0.+train_train1')
  import pdb; pdb.set_trace()
