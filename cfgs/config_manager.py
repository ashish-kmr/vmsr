from src import utils
import numpy as np
import logging

class ConfigManager():
  def __init__(self):
    # List of def_args in sequence.
    self.def_args = []
    self.name_args = []
    self.compile()
  
  def split_string(self, string, tokens):
    ls = [string]
    for t in tokens:
      ls_ = []
      for l in ls: 
        ls_ += l.split(t)
      ls = ls_
    return ls

  def process_string(self, string):
    str_parts = self.split_string(string, '.+')
    assert(len(str_parts) == len(self.def_args)), \
      'len(str_parts) = {:d}, len(self.def_args) = {:d}'.\
      format(len(str_parts), len(self.def_args))
    assert(len(self.name_args) == len(self.def_args))
    args = utils.Foo()
    args = self.pre_hook(args)
    for n, d, p in zip(self.name_args, self.def_args, str_parts):
      args = getattr(self, '_'+n+'_hook')(d, p, args)
    args = self.post_hooks(args)
    logging.error('args: %s', args)
    return args 
  
  def post_hooks(self, args):
    return args
  
  def pre_hook(self, args):
    return args
  
  def get_default_string(self):
    return '.'.join([x.get_default_string() for x in self.def_args])

  def compile(self):
    """Code for compiling things so that we can return a complete string."""
    pass

def str_to_float(x):
  return float(x.replace('x', '.').replace('n', '-'))

class ConfigManagerSolver(ConfigManager):
  def _solver_setup(self):
    args = [('dlw', 'dlw20'), ('rlw', 'rlw1'), ('elw', 'elw1en0'),
      ('init_lr', 'lr1en3'), ('typ', 'adam2'), ('maxiter', 'MI6e4'),
      ('stepiter', 'SI2e4'), ('clip', 'noclip'), ('seed', 'seed0')]
    args = utils.DefArgs(args)
    return args
  
  def _solver_hook(self, solver_def_args, solver_str, args):
    solver = utils.Foo(
      seed=None, learning_rate_decay=None, clip_gradient_norm=None, max_steps=None,
      initial_learning_rate=None, momentum=None, steps_per_decay=None,
      logdir=None, sync=False, adjust_lr_sync=True, wt_decay=0.0001,
      data_loss_wt=None, reg_loss_wt=None, freeze_conv=False, num_workers=1,
      task=0, ps_tasks=0, master='', typ=None, momentum2=None,
      adam_eps=1e-8)

    solver_vars = solver_def_args.process_string(solver_str)
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
    args.solver = solver
    return args 

def test_config_manager_string():
  class TestConfigManager(ConfigManager):
    def _solver_setup(self):
      args = [('dlw', 'dlw20'), ('rlw', 'rlw1'), ('elw', 'elw1en0'),
        ('init_lr', 'lr1en3'), ('typ', 'adam2'), ('maxiter', 'MI6e4'),
        ('stepiter', 'SI2e4'), ('clip', 'noclip'), ('seed', 'seed0')]
      args = utils.DefArgs(args)
      return args

    def _solver_hook(self, solver_def_args, solver_str):
      args = solver_def_args.process_string(solver_str)
      return args

    def compile(self):
      names = ['solver']
      for n in names:
        self.def_args += [getattr(self, '_'+n+'_setup')()]
        self.name_args += [n]
  
  cm = TestConfigManager()
  cm.compile()
  print(cm.get_default_string())
  print(cm.process_string(''))

def test_config_manager_solver_string():
  class TestConfigManager(ConfigManagerSolver):
    def compile(self):
      names = ['solver']
      for n in names:
        self.def_args += [getattr(self, '_'+n+'_setup')()]
        self.name_args += [n]
  
  cm = TestConfigManager()
  cm.compile()
  import pdb; pdb.set_trace()
  print(cm.get_default_string())
  print(cm.process_string(''))
