import os, sys, numpy as np, logging, tensorflow as tf
from src import utils
from cfgs import config_common as cc
from env import toy_graph_env as env 

str_to_float = cc.str_to_float

def get_default_args():
  summary_args = utils.Foo(display_iters=100, test_iters=200,
    arop_full_summary_iters=200)
  control_args = utils.Foo(train=False, test=False, reset_rng_seed=False,
    only_eval_when_done=False, test_mode=None, name=None)
  return summary_args, control_args

def get_default_arch_args():
  arch_args = utils.Foo(embedding_dim=32, distance_dim=32, num_nodes=16,
    graph_iters=16, factored_distance_matrix=False)
  return arch_args

def get_odotask_vars(arch_str):
  if arch_str == '': vals = []
  else: vals = arch_str.split('_')
  ks = ['batch_size', 'randomize']
  ks = ks[:len(vals)]
  
  # Different settings.
  if len(vals) == 0: ks.append('batch_size'); vals.append('4')
  if len(vals) == 1: ks.append('randomize'); vals.append('0')
  assert(len(vals) == 2)
  
  vars = utils.Foo()
  for k, v in zip(ks, vals):
    setattr(vars, k, v)

  logging.error('odotask_vars: %s', vars)
  return vars

def process_odotask_str(odotask_str):
  odotask_vars = get_odotask_vars(odotask_str)
  batch_size = int(odotask_vars.batch_size)
  randomize = int(odotask_vars.randomize) > 0
  graph_factory_task_params = env.get_graph_factory_task_params(
    batch_size=batch_size, randomize_graph=randomize)

  odotask = utils.Foo()
  odotask.seed = 0
  odotask.graph_factory_task_params = graph_factory_task_params
  return odotask

def get_arch_vars(arch_str):
  if arch_str == '': vals = []
  else: vals = arch_str.split('_')
  ks = ['num_nodes', 'distance_dim', 'embedding_dim', 'graph_iters',
    'factored', 'direct_embedding']
  ks = ks[:len(vals)]
  
  if len(vals) == 0: ks.append('num_nodes'); vals.append('N16')
  if len(vals) == 1: ks.append('distance_dim'); vals.append('D1')
  if len(vals) == 2: ks.append('embedding_dim'); vals.append('E32')
  if len(vals) == 3: ks.append('graph_iters'); vals.append('I16')
  if len(vals) == 4: ks.append('factored'); vals.append('F0')
  if len(vals) == 5: ks.append('direct_embedding'); vals.append('DE0')
  assert(len(vals) == 6)

  vars = utils.Foo()
  for k, v in zip(ks, vals):
    setattr(vars, k, v)

  logging.error('arch_vars: %s', vars)
  return vars

def process_arch_str(args, arch_str):
  # This function modifies args.
  args.arch = get_default_arch_args()
  arch_vars = get_arch_vars(arch_str)
  
  args.arch.num_nodes = int(arch_vars.num_nodes[1:])
  args.arch.distance_dim = int(arch_vars.distance_dim[1:])
  args.arch.graph_iters = int(arch_vars.graph_iters[1:])
  args.arch.factored_distance_matrix = int(arch_vars.factored[1:]) > 0
  args.arch.embedding_dim = int(arch_vars.embedding_dim[1:])
  args.arch.direct_embedding = int(arch_vars.direct_embedding[2:]) > 0

  # Copy relevant parameters from task_params needed to construct the network. 
  task_params = utils.Foo()
  task_params.batch_size = args.odotask.graph_factory_task_params.batch_size
  for s in ['batch_size', 'num_sources', 'num_targets']:
    setattr(task_params, s, getattr(args.odotask.graph_factory_task_params, s))
  task_params.state_dim = 2
  args.arch.task_params = task_params
  return args

def get_mode_vars(mode_str):
  if mode_str == '': vals = []; 
  else: vals = mode_str.split('_')
  ks = ['mode']
  ks = ks[:len(vals)]

  if len(vals) == 0: ks.append('mode');  vals.append('')
  assert(len(vals) == 1)
  
  vars = utils.Foo()
  for k, v in zip(ks, vals):
    setattr(vars, k, v)
  logging.error('mode_vars: %s', vars)
  return vars

def adjust_args_for_mode(args, mode_vars):
  mode = mode_vars.mode
  if mode == 'train':
    args.control.train = True
  else:
    args.control.test = True
  args.odotask.graph_factory_task_params.mode = mode
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
  mode_vars = get_mode_vars(mode_str)
  args = adjust_args_for_mode(args, mode_vars)

  args.control.name = '{:s}'.format(mode_str)
  if mode_vars.mode == 'val':
    args.control.test_mode = 'val'
  args.env_class = env.GraphFactory

  # Log the arguments
  logging.error('%s', args)
  return args
