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
  summary_args = utils.Foo(display_iters=20, test_iters=26,
    arop_full_summary_iters=14)
  control_args = utils.Foo(train=False, test=False, reset_rng_seed=False,
    only_eval_when_done=False, test_mode=None, name=None)
  return summary_args, control_args

def get_default_arch_args():
  batch_norm_param = {'center': True, 'scale': True,
    'activation_fn':tf.nn.relu}
  arch_args = utils.Foo(encoder='resnet_v2_50', fc_neurons=[256, 256], 
    batch_norm_param=batch_norm_param)
  return arch_args

def get_odotask_vars(arch_str):
  if arch_str == '': vals = []
  else: vals = arch_str.split('_')
  ks = ['noise']
  ks = ks[:len(vals)]
  
  # Different settings.
  if len(vals) == 0: ks.append('noise'); vals.append('N1en1')

  assert(len(vals) == 1)

  vars = utils.Foo()
  for k, v in zip(ks, vals):
    setattr(vars, k, v)

  logging.error('odotask_vars: %s', vars)
  return vars

def process_odotask_str(odotask_str):
  odotask_vars = get_odotask_vars(odotask_str)
  noise = str_to_float(odotask_vars.noise[1:])
  task_params = nne.get_default_args_top_view_env(noise=noise, fovs=[128], 
    view_scales=[0.25], batch_size=1, ignore_roads=True)
  odotask = utils.Foo()
  odotask.task_params = task_params
  odotask.seed = 0
  return odotask

def get_arch_vars(arch_str):
  if arch_str == '': vals = []
  else: vals = arch_str.split('_')
  ks = ['var1', 'var2', 'var3']
  ks = ks[:len(vals)]
  
  # Exp Ver.
  if len(vals) == 0: ks.append('var1'); vals.append('v0')
  # custom arch.
  if len(vals) == 1: ks.append('var2'); vals.append('')
  # map scape for projection baseline.
  if len(vals) == 2: ks.append('var3'); vals.append('fr2')

  assert(len(vals) == 3)

  vars = utils.Foo()
  for k, v in zip(ks, vals):
    setattr(vars, k, v)

  logging.error('arch_vars: %s', vars)
  return vars

def process_arch_str(args, arch_str):
  # This function modifies args.
  args.arch = get_default_arch_args()
  arch_vars = get_arch_vars(arch_str)
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
  args.env_class = nne.EnvMultiplexer

  # Log the arguments
  logging.error('%s', args)
  return args
