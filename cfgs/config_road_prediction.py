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
  summary_args = utils.Foo(display_iters=20, test_iters=8,
    arop_full_summary_iters=1)
  control_args = utils.Foo(train=False, test=False, reset_rng_seed=False,
    only_eval_when_done=False, test_mode=None, name=None)
  return summary_args, control_args

def get_default_arch_args():
  arch_args = utils.Foo(encoder='resnet_v2_small', batch_norm_param=None)
  return arch_args

def get_odotask_vars(arch_str):
  if arch_str == '': vals = []
  else: vals = arch_str.split('_')
  ks = ['bs', 'dilate']
  ks = ks[:len(vals)]
  
  # Different settings.
  if len(vals) == 0: ks.append('bs'); vals.append('128')
  if len(vals) == 1: ks.append('dilate'); vals.append('0')

  assert(len(vals) == 2)

  vars = utils.Foo()
  for k, v in zip(ks, vals):
    setattr(vars, k, v)

  logging.error('odotask_vars: %s', vars)
  return vars

def process_odotask_str(odotask_str):
  odotask_vars = get_odotask_vars(odotask_str)
  batch_size = int(odotask_vars.bs)
  dilate_size = int(odotask_vars.dilate)
  top_view_task_params = nne.get_top_view_env_task_params(noise=0,
    fovs=[128], view_scales=[0.25], batch_size=1, ignore_roads=False, 
    output_roads=True, road_dilate_disk_size=dilate_size)
  road_prediction_task_params = nne.get_road_prediction_task_params(
    batch_size=batch_size, add_flips=True) 
  odotask = utils.Foo()
  odotask.seed = 0
  odotask.top_view_task_params = top_view_task_params 
  odotask.road_prediction_task_params = road_prediction_task_params 
  return odotask

def get_arch_vars(arch_str):
  if arch_str == '': vals = []
  else: vals = arch_str.split('_')
  ks = ['encoder', 'unet']
  ks = ks[:len(vals)]
  
  # Encoder
  if len(vals) == 0: ks.append('encoder'); vals.append('rssmall')
  if len(vals) == 1: ks.append('unet'); vals.append('0')

  assert(len(vals) == 2)

  vars = utils.Foo()
  for k, v in zip(ks, vals):
    setattr(vars, k, v)

  logging.error('arch_vars: %s', vars)
  return vars

def process_arch_str(args, arch_str):
  # This function modifies args.
  args.arch = get_default_arch_args()
  arch_vars = get_arch_vars(arch_str)
  if arch_vars.encoder == 'rssmall':
    args.arch.encoder = 'resnet_v2_small'
  elif arch_vars.encoder == 'small':
    args.arch.encoder = 'simple_net'
    args.arch.batch_norm_param = None
  else:
    logging.fatal('arch_vars.encoder is not rssmall, simple_net')
    assert(False)

  if (arch_vars.unet) > 0:
    args.arch.unet = True
  else:
    args.arch.unet = False
  
  # Copy relevant parameters from task_params needed to construct the network. 
  task_params = utils.Foo()
  task_params.batch_size = args.odotask.road_prediction_task_params.batch_size
  task_params.fovs = args.odotask.top_view_task_params.fovs
  task_params.relight = False; task_params.relight_fast = False;
  task_params.num_classes = 2
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
  mode, imset = mode_str.split('.')
  args = cc.adjust_args_for_mode(args, mode)
  args.odotask.dataset = factory.get_dataset('campus', imset)
  args.odotask.names = args.odotask.dataset.get_imset()

  args.control.name = '{:s}_on_{:s}'.format(mode, imset)
  args.env_class = nne.RoadMultiplexer

  # Log the arguments
  logging.error('%s', args)
  return args
