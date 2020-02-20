import os
import numpy as np
import logging
from src import utils

def adjust_args_for_mode(args, mode_vars):
  mode = mode_vars.mode
  k = str_to_float(mode_vars.x_steps[:-1])
  args.arch.num_steps = int(k*args.arch.num_steps)
  
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
  return args

def get_mode_vars(mode_str):
  args = [('mode', 'train'), ('imset', 'train'), ('x_steps', '1X')]
  mode_args = utils.DefArgs(args)
  mode_vars = mode_args.process_string(mode_str)
  logging.error('mode_vars: %s', mode_vars)
  return mode_vars
 
# def get_mode_vars(mode_str):
#   if mode_str == '': vals = []; 
#   else: vals = mode_str.split('_')
#   ks = ['mode', 'imset', 'x_steps']
#   ks = ks[:len(vals)]
# 
#   if len(vals) == 0: ks.append('mode');  vals.append('')
#   if len(vals) == 1: ks.append('imset');  vals.append('')
#   if len(vals) == 2: ks.append('x_steps');  vals.append('1X')
# 
#   assert(len(vals) == 3)
#   
#   vars = utils.Foo()
#   for k, v in zip(ks, vals):
#     setattr(vars, k, v)
#   logging.error('mode_vars: %s', vars)
#   return vars
# 
# def get_solver_vars(solver_str):
#   if solver_str == '': vals = []; 
#   else: vals = solver_str.split('_')
#   ks = ['dlw', 'rlw', 'elw', 'init_lr', 'typ', 'maxiter', 'stepiter', 'clip', 'seed']
#   ks = ks[:len(vals)]
# 
#   # data loss weight.
#   if len(vals) == 0: ks.append('dlw');  vals.append('dlw20')
#   # reg loss wt
#   if len(vals) == 1: ks.append('rlw');  vals.append('rlw1')
#   # entropy loss wt
#   if len(vals) == 2: ks.append('elw');  vals.append('elw1en0')
#   # init lr 
#   if len(vals) == 3: ks.append('init_lr');  vals.append('lr1en3')
#   # Adam
#   if len(vals) == 4: ks.append('typ');  vals.append('adam2')
#   # how long to train for.
#   if len(vals) == 5: ks.append('maxiter');  vals.append('MI6e4')
#   # init lr
#   if len(vals) == 6: ks.append('stepiter');  vals.append('SI2e4')
#   # Gradient clipping or not.
#   if len(vals) == 7: ks.append('clip'); vals.append('noclip');
#   # Solver seed.
#   if len(vals) == 8: ks.append('seed'); vals.append('seed0');
# 
#   assert(len(vals) == 9)
#   
#   vars = utils.Foo()
#   for k, v in zip(ks, vals):
#     setattr(vars, k, v)
#   logging.error('solver_vars: %s', vars)
#   return vars

def str_to_float(x):
  return float(x.replace('x', '.').replace('n', '-'))

def process_solver_str(solver_str):
  solver = utils.Foo(
      seed=None, learning_rate_decay=None, clip_gradient_norm=None, max_steps=None,
      initial_learning_rate=None, momentum=None, steps_per_decay=None,
      logdir=None, sync=False, adjust_lr_sync=True, wt_decay=0.0001,
      data_loss_wt=None, reg_loss_wt=None, freeze_conv=False, num_workers=1,
      task=0, ps_tasks=0, master='', typ=None, momentum2=None,
      adam_eps=1e-8)

  args = [('dlw', 'dlw20'), ('rlw', 'rlw1'), ('elw', 'elw1en0'), ('init_lr',
    'lr1en3'), ('typ', 'adam2'), ('maxiter', 'MI6e4'), ('stepiter', 'SI2e4'),
    ('clip', 'noclip'), ('seed', 'seed0')]
  solver_args = utils.DefArgs(args)
  solver_vars = solver_args.process_string(solver_str)
  logging.error('solver_vars: %s', solver_vars)
  
  # Clobber with overrides from solver str.
  solver.data_loss_wt          = str_to_float(solver_vars.dlw[3:])
  solver.reg_loss_wt           = str_to_float(solver_vars.rlw[3:])
  solver.ent_loss_wt           = str_to_float(solver_vars.elw[3:])
  solver.initial_learning_rate = str_to_float(solver_vars.init_lr[2:])
  solver.steps_per_decay       = int(str_to_float(solver_vars.stepiter[2:]))
  solver.max_steps             = int(str_to_float(solver_vars.maxiter[2:]))
  solver.seed                  = int(str_to_float(solver_vars.seed[4:]))

  clip = solver_vars.clip
  if clip == 'noclip' or clip == 'nocl':
    solver.clip_gradient_norm = 0
  elif clip[:4] == 'clip':
    solver.clip_gradient_norm = str_to_float(clip[4:])
  else:
    logging.fatal('Unknown solver_vars.clip: %s', clip)
    assert(False)

  typ = solver_vars.typ
  if typ == 'adam':
    solver.typ = 'adam'
    solver.momentum = 0.9
    solver.momentum2 = 0.999
    solver.learning_rate_decay = 1.0
  elif typ == 'adam2':
    solver.typ = 'adam'
    solver.momentum = 0.9
    solver.momentum2 = 0.999
    solver.learning_rate_decay = 0.1
  elif typ == 'sgd':
    solver.typ = 'sgd'
    solver.momentum = 0.99
    solver.momentum2 = None
    solver.learning_rate_decay = 0.1
  else:
    logging.fatal('Unknown solver_vars.typ: %s', typ)
    assert(False)

  logging.error('solver: %s', solver)
  return solver
