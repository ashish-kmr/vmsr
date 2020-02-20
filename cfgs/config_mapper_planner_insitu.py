import os, sys
import numpy as np
import logging
import tensorflow as tf
from src import utils
from cfgs import config_common as cc
from env import mapper_env as me 
from env import toy_landmark_env as tle
from env import mp_env
from env import factory
from cfgs import config_manager as cm
from cfgs import config_map_follower as cmf
str_to_float = cm.str_to_float

def get_default_args():
  summary_args = utils.Foo(display_iters=20, test_iters=40,
    arop_full_summary_iters=8, summary_iters=20)
  control_args = utils.Foo(train=False, test=False, reset_rng_seed=False,
    only_eval_when_done=False, test_mode=None, name=None)
  return summary_args, control_args

def get_default_arch_args():
  batch_norm_param = {'center': True, 'scale': True,
    'activation_fn':tf.nn.relu}
  arch_args = utils.Foo(batch_norm_param=batch_norm_param, sample_gt_prob='zero')
  return arch_args

class ConfigManager(cmf.ConfigManager):
  def _mode_setup(self):
    args = [('mode', 'train'), ('imset', 'train'), ('for_test', '0'), ('max_for_test', '0')]
    return utils.DefArgs(args)

  def _mode_hook(self, mode_args, mode_str, args):
    mode_vars = mode_args.process_string(mode_str)
    mode = mode_vars.mode
    if mode == 'train':
      args.control.train = True
      args.control.mode = 'train'
      # m.sample_action_type
    elif mode == 'val1':
      args.control.test = True
      args.control.mode = 'val'
    elif mode == 'val2':
      args.control.test = True
      args.control.mode = 'live'
    elif mode == 'val3':
      # For visualizing map as a function of the number of images
      args.control.test = True
      args.arch.cumsum = True
      args.control.mode = 'live'
      args.summary.test_iters = 100
      args.task.env_task_params_2.add_flips = True
    elif mode == 'bench':
      args.control.test = True
      args.control.mode = 'live'
      args.control.only_eval_when_done = True
      args.summary.test_iters = 1000
      args.control.reset_rng_seed = True
    else:
      logging.fatal('Unknown mode: %s.', mode)
      assert(False)
    args.task.dataset = factory.get_dataset(args.task.dataset_name, mode_vars.imset)
    args.task.names = args.task.dataset.get_imset()
    args.control.name = '{:s}'.format(mode_str)
    
    args.control.for_test = int(mode_vars.for_test)
    args.control.max_for_test = int(mode_vars.max_for_test)
    
    args.task.dataset = factory.get_dataset(args.task.dataset_name, mode_vars.imset)
    args.task.names = args.task.dataset.get_imset()
    args.control.name = '{:s}'.format(mode_str)
    return args
 
  def _arch_setup(self):
    args_ = [('ver', 'v0'), ('image_cnn', 'rs18'), ('freeze_conv', 'frz1'), 
      ('batch_norm', 'bn1'), ('dim_reduce_neurons', 'dr0'), ('combine_type', 'add'), 
      ('share_wts', 'sh0')]
    return utils.DefArgs(args_)

  def _arch_hook(self, arch_args, arch_str, args):
    arch_vars = arch_args.process_string(arch_str)
    arch = get_default_arch_args()
    logging.error('arch_vars: %s', arch_vars)
    
    arch.ver = arch_vars.ver
    arch.batch_norm = int(arch_vars.batch_norm[2:]) > 0
    arch.image_cnn = arch_vars.image_cnn
    arch.freeze_conv = int(arch_vars.freeze_conv[3:]) > 0
    arch.dim_reduce_neurons = int(arch_vars.dim_reduce_neurons[2:])
    arch.combine_type = arch_vars.combine_type
    arch.cumsum = False
    arch.share_wts = int(arch_vars.share_wts[2:]) > 0
    arch.share_imgs = False
    arch.use_goal_imgs = False

    assert(arch.ver in ['v0', 'v1', 'v2', 'v3'])

    if arch.ver == 'v2' or arch.ver == 'v3':
      arch.share_imgs = True
      arch.use_goal_imgs = True

    if arch.image_cnn == 'rs50':
      arch.image_cnn = 'resnet_v2_50'
    elif arch.image_cnn == 'rs18':
      arch.image_cnn = 'resnet_v2_18'
    elif arch.image_cnn == 'sn':
      arch.image_cnn = 'simple_net'
    else:
      assert(False), 'image_cnn is not defined.'
    
    args.arch = arch
    return args

  def compile(self):
    names = ['task', 'arch', 'solver', 'mode']
    for n in names:
      self.def_args += [getattr(self, '_'+n+'_setup')()]
      self.name_args += [n]

  def post_hooks(self, args):
    args.task.env_task_params_2.num_samples = args.task.env_task_params_2.mapping_samples
    extent_samples = args.task.env_task_params_2.extent_samples
    base_resolution = 0.125
    fovs = [int(extent_samples*4/5*base_resolution)+1]
    vs = [base_resolution]
    args.task.env_task_params.fovs = fovs
    args.task.env_task_params.vs = vs
    
    if args.control.max_for_test > 0:
      num_samples = args.control.max_for_test
      logging.error('Changing num_samples to: %d', num_samples)
      args.task.env_task_params_2.num_samples = num_samples
    return args

  def pre_hook(self, args):
    args.summary, args.control = get_default_args()
    return args

def test_get_args_for_config():
  # args1 = get_args_for_config('..+train_train1')
  cm_tf = ConfigManager()
  import pdb; pdb.set_trace()
  print(cm_tf.get_default_string())
  args1 = cm_tf.process_string('..+train_train1')
  args2 = cm_tf.process_string('bs4_N1en1_128_4_20_br1x0_rng_sbpd.v0_ns80_128_1_gru_isd100_bn0_dnc0_0.+train_train1')
