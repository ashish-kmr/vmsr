import os, sys
import numpy as np
import logging
import tensorflow as tf
from src import utils
from cfgs import config_common as cc
from env import noisy_nav_env as nne
from env import factory

str_to_float = cc.str_to_float

def get_default_args():
  summary_args = utils.Foo(display_iters=1, test_iters=8,
    arop_full_summary_iters=8)
  control_args = utils.Foo(train=False, test=False, reset_rng_seed=False,
    only_eval_when_done=False, test_mode=None, name=None)
  return summary_args, control_args

def get_default_arch_args():
  batch_norm_param = {'center': True, 'scale': True,
    'activation_fn':tf.nn.relu}
  arch_args = utils.Foo(encoder='resnet_v2_small', fc_neurons=[256, 256], 
    batch_norm_param=batch_norm_param, num_steps=20, sample_gt_prob='zero')
  return arch_args

def get_odotask_vars(arch_str):
  if arch_str == '': vals = []
  else: vals = arch_str.split('_')
  ks = ['noise', 'toc', 'batch_size', 'flips', 'gtD', 'dil', 'history',
    'teacher_steps', 'minD', 'maxD', 'base_resolution', 'rejection_sampling']
  ks = ks[:len(vals)]
  
  # Different settings.
  if len(vals) == 0: ks.append('noise'); vals.append('N1en1')
  if len(vals) == 1: ks.append('toc'); vals.append('0')
  if len(vals) == 2: ks.append('batch_size'); vals.append('4')
  if len(vals) == 3: ks.append('flips'); vals.append('0')
  if len(vals) == 4: ks.append('gtD'); vals.append('1')
  if len(vals) == 5: ks.append('dil'); vals.append('dil0')
  if len(vals) == 6: ks.append('history'); vals.append('h0')
  if len(vals) == 7: ks.append('teacher_steps'); vals.append('ts0')
  if len(vals) == 8: ks.append('minD'); vals.append('25')
  if len(vals) == 9: ks.append('maxD'); vals.append('200')
  if len(vals) == 10: ks.append('base_resolution'); vals.append('br1x0')
  if len(vals) == 11: ks.append('rejection_sampling'); vals.append('nors')

  assert(len(vals) == 12)

  vars = utils.Foo()
  for k, v in zip(ks, vals):
    setattr(vars, k, v)

  logging.error('odotask_vars: %s', vars)
  return vars

def process_odotask_str(odotask_str):
  odotask_vars = get_odotask_vars(odotask_str)
  noise = str_to_float(odotask_vars.noise[1:])
  terminate_on_complete = int(odotask_vars.toc) > 0 
  batch_size = int(odotask_vars.batch_size)
  add_flips = int(odotask_vars.flips) > 0
  gt_delta_to_goal = int(odotask_vars.gtD) > 0
  dilate_radius = int(odotask_vars.dil[3:])
  history = int(odotask_vars.history[1:])
  teacher_steps = int(odotask_vars.teacher_steps[2:])
  min_dist = str_to_float(odotask_vars.minD)
  max_dist = str_to_float(odotask_vars.maxD)
  base_resolution = str_to_float(odotask_vars.base_resolution[2:])
  dilate_radius = int(np.round(dilate_radius * base_resolution))
  rejection_sampling = False if odotask_vars.rejection_sampling == 'nors' else True

  fovs = int(np.round(128))
  vs = 0.25 / base_resolution
  
  top_view_task_params = nne.get_top_view_env_task_params(noise=noise,
    fovs=[fovs], view_scales=[vs], batch_size=batch_size, ignore_roads=False, 
    output_roads=True, road_dilate_disk_size=dilate_radius, map_max_size=None,
    base_resolution=base_resolution)
  
  follower_task_params = nne.get_follower_task_params(batch_size=batch_size, 
    gt_delta_to_goal=gt_delta_to_goal, terminate_on_complete=terminate_on_complete, 
    compute_optimal_actions=False, add_flips=add_flips, history=history,
    teacher_steps=teacher_steps, min_dist=min_dist, max_dist=max_dist,
    rejection_sampling=rejection_sampling)

  odotask = utils.Foo()
  odotask.seed = 0
  odotask.top_view_task_params = top_view_task_params 
  odotask.follower_task_params = follower_task_params
  return odotask

def get_arch_vars(arch_str):
  if arch_str == '': vals = []
  else: vals = arch_str.split('_')
  ks = ['ns', 'encoder', 'eps_greedy', 'loss_type', 'isd', 'ver', 'eblw']
  ks = ks[:len(vals)]
  
  # Exp Ver.
  if len(vals) == 0: ks.append('ns'); vals.append('ns20')
  # Encoder
  if len(vals) == 1: ks.append('encoder'); vals.append('rssmall')
  # Epsilon Greedy 
  if len(vals) == 2: ks.append('eps_greedy'); vals.append('0')
  # a2c loss or supervised training
  if len(vals) == 3: ks.append('loss_type'); vals.append('a2c')
  # entropy batch loss wt
  if len(vals) == 4: ks.append('isd'); vals.append('isd100')
  # entropy batch loss wt
  if len(vals) == 5: ks.append('ver'); vals.append('v0')
  # entropy batch loss wt
  if len(vals) == 6: ks.append('eblw'); vals.append('0')

  assert(len(vals) == 7)

  vars = utils.Foo()
  for k, v in zip(ks, vals):
    setattr(vars, k, v)

  logging.error('arch_vars: %s', vars)
  return vars

def process_arch_str(args, arch_str):
  # This function modifies args.
  args.arch = get_default_arch_args()
  arch_vars = get_arch_vars(arch_str)
  args.arch.num_steps = int(arch_vars.ns[2:])
  if arch_vars.encoder == 'rssmall':
    args.arch.encoder = 'resnet_v2_small'
  elif arch_vars.encoder == 'small':
    args.arch.encoder = 'simple_net'
    args.arch.batch_norm_param = None
  elif arch_vars.encoder == 'smallBN':
    args.arch.encoder = 'simple_net_bn'
  else:
    logging.fatal('arch_vars.encoder is not rssmall, simple_net')
    assert(False)
  args.arch.eps_greedy = str_to_float(arch_vars.eps_greedy)
  args.arch.entb_loss_wt = str_to_float(arch_vars.eblw)
  args.arch.loss_type = arch_vars.loss_type
  
  if args.arch.loss_type == 'sup':
    logging.error('Supervised training of policy, modifying odotask params.')
    args.odotask.follower_task_params.compute_optimal_actions = True 
    args.odotask.top_view_task_params.map_max_size = 4000
    args.arch.sample_gt_prob = arch_vars.isd
  
  elif args.arch.loss_type == 'a2c':
    args.arch.sample_gt_prob = 'zero'
  
  # Copy relevant parameters from task_params needed to construct the network. 
  task_params = utils.Foo()
  task_params.batch_size = args.odotask.follower_task_params.batch_size
  task_params.fovs= args.odotask.top_view_task_params.fovs
  task_params.num_actions = len(args.odotask.follower_task_params.act_f) + 2*len(args.odotask.follower_task_params.act_r)
  task_params.relight = False; task_params.relight_fast = False;
  task_params.use_roads = args.odotask.top_view_task_params.outputs.top_view_roads 
  task_params.teacher_steps = args.odotask.follower_task_params.teacher_steps
  
  task_params.use_last_action = False
  args.arch.use_teacher = False

  if args.odotask.follower_task_params.gt_delta_to_goal:
    task_params.use_gt_delta_to_goal = True
    args.arch.compute_egomotion = False 
    args.arch.use_views_for_egomotion = False
  else:
    task_params.use_gt_delta_to_goal = False
    args.arch.compute_egomotion = True
    args.arch.use_views_for_egomotion = False
    if arch_vars.ver == 'v0':
      None
    elif arch_vars.ver == 'v1':
      args.arch.use_views_for_egomotion = True
    elif arch_vars.ver == 'v2':
      args.arch.use_teacher = True
      args.arch.use_views_for_egomotion = True
    else:
      logging.error('arch_vars.ver is not v0, v1 or v2.')
      assert(False)
    task_params.use_last_action = True
  
  task_params.history = args.odotask.follower_task_params.history
  
  args.arch.task_params = task_params
  return args

def get_args_for_config(config_name):
  args = utils.Foo()
  args.summary, args.control = get_default_args()
  
  exp_name, mode_str = config_name.split('+')
  arch_str, solver_str, odotask_str = exp_name.split('.')
  logging.error('config_name: %s', config_name)
  logging.error('arch_str: %s', arch_str)
  logging.error('odotask_str: %s', odotask_str)
  logging.error('solver_str: %s', solver_str)
  logging.error('mode_str: %s', mode_str)

  args.solver = cc.process_solver_str(solver_str)
  args.odotask = process_odotask_str(odotask_str)

  args = process_arch_str(args, arch_str)

  # Train, test, etc.
  mode_vars = cc.get_mode_vars(mode_str)
  args = cc.adjust_args_for_mode(args, mode_vars)
  args.odotask.dataset = factory.get_dataset('campus', mode_vars.imset)
  args.odotask.names = args.odotask.dataset.get_imset()

  args.control.name = '{:s}'.format(mode_str)
  args.env_class = nne.EnvMultiplexer

  # Log the arguments
  logging.error('%s', args)
  return args
