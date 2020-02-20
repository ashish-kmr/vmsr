import logging, os, numpy as np
from src import utils

class Env():
  """Has functions: get_common_data(),
  pre_common_data(), get_features(), get_optimal_action(), take_action(),
  get_features_name(), pre_features()."""
  def __init__(self, args):
    self.num_symbols = args.task_params.num_actions
    self.num_patterns = args.task_params.num_patterns
    self.num_steps = args.task_params.num_steps
    self.batch_size = args.task_params.batch_size
    self.type = args.task_params.type

    if self.type == 'text':
      from env import text_env as te
      t = te.TextLoader('text-data/', self.num_patterns, self.num_steps, encoding='utf-8')
      t.create_batches()
      b = t.next_batch()[0]
      self.text = b
    
    elif self.type == 'maps':
      # Load random trajectories from maps as we do for the actual training.
      from env import toy_landmark_env as tle
      from cfgs import config_trajectory_follower as config
      imset = 'train1'
      args = config.get_args_for_config('v0_ns{:d}_256_1_gru_isd415_bn1.dlw1e0_rlw1e0_ent0e0_lr1en4____clip2.bs1_N0_{:d}+train_{:s}'.format(self.num_steps, self.num_steps, imset))
      rng = np.random.RandomState(0)
      oo = tle.EnvMultiplexer(args.task, 0, 1)
      e = oo.sample_env(rng)
      text = []
      for i in range(self.num_patterns):
        e.reset(rng)
        tmp = e.get_common_data()
        text.append(tmp['teacher_actions'])
      self.text = np.concatenate(text, axis=0)
    logging.info('')
  
  def reset(self, rng):
    es = []
    for i in range(self.batch_size):
      if self.type == 'random':
        rng_ = np.random.RandomState(rng.randint(self.num_patterns))
        e = rng_.randint(self.num_symbols, size=(1, self.num_steps))
      elif self.type == 'text' or self.type == 'maps':
        id_ = rng.randint(self.num_patterns)
        e = self.text[id_,np.newaxis,:]
      es.append(e)
    self.english = np.concatenate(es, 0)
    if self.type != 'maps':
      self.french = self.english[:,::-1]*1
    else:
      self.french = self.english*1
    self.pos = 0
    return [self.pos]*self.batch_size
  def get_common_data(self):
    return {'english': self.english}
  def pre_common_data(self, _):
    return _ 
  def get_features(self, state, j):
    return {}
  def pre_features(self, _):
    return _ 
  def get_optimal_action(self, state, j):
    acts = self.french[:,self.pos,np.newaxis] == np.arange(self.num_symbols).reshape((1,-1))
    return acts*1.
  def get_targets(self, state, j):
    acts = self.get_optimal_action(state, j)
    acts = np.expand_dims(acts, axis=1)*1
    return {'gt_action': acts}
  def take_action(self, state, _, j):
    self.pos = self.pos+1
    return [self.pos]*self.batch_size, [0]*self.batch_size
  def get_gamma(self):
    return 0.99

class Obj():
  """Has the function sample_env(). This function returns an environment for
  training."""
  def __init__(self, args):
    self.env = Env(args)
  def sample_env(self, rng):
    return self.env


