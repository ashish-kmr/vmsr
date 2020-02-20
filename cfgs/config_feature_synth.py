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

class ConfigManager(cm.ConfigManagerSolver):
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
      args.control.test = True
      args.control.mode = 'live'
      args.task.env_task_params_2.add_flips = False
      args.summary.test_iters = 800
      args.task.env_task_params_2.repeat_map_samples = args.summary.test_iters
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
    return args
 
  def _arch_setup(self):
    args_ = [('ver', 'v0'), ('image_cnn', 'rs18'), ('freeze_conv', 'frz1'),
      ('batch_norm', 'bn1'), ('dim_reduce_neurons', 'dr0'), ('combine_type',
      'add'), ('test_f', '0x25'), ('dropout', 'do0'), ('combine_as', 'conv'), 
      ('bn_pose', 'bnp1'), ('rng_per_test', 'rngt50')]
    return utils.DefArgs(args_)

  def _arch_hook(self, arch_args, arch_str, args):
    arch_vars = arch_args.process_string(arch_str)
    arch = get_default_arch_args()
    logging.error('arch_vars: %s', arch_vars)
    
    arch.use_bn = int(arch_vars.batch_norm[2:]) > 0
    arch.image_cnn = arch_vars.image_cnn
    arch.freeze_conv = int(arch_vars.freeze_conv[3:]) > 0
    arch.dim_reduce_neurons = int(arch_vars.dim_reduce_neurons[2:])
    assert(arch_vars.combine_type in ['gru', 'max', 'wtadd', 'add'])
    arch.combine_type = arch_vars.combine_type
    arch.combine_as = arch_vars.combine_as
    arch.test_f = str_to_float(arch_vars.test_f)
    arch.bn_pose = int(arch_vars.bn_pose[3:]) > 0
    arch.use_dropout = int(arch_vars.dropout[2:]) > 0
    arch.rng_per_test = int(arch_vars.rng_per_test[4:])
      
    assert(arch.combine_as in ['conv', 'fc', 'fcconv'])
    if arch.image_cnn == 'rs50':
      arch.image_cnn = 'resnet_v2_50'
    elif arch.image_cnn == 'rs18':
      arch.image_cnn = 'resnet_v2_18'
    elif arch.image_cnn == 'sn':
      arch.image_cnn = 'simple_net'
    elif arch.image_cnn == 'sn5':
      arch.image_cnn = 'simple_net_5'
    else:
      assert(False), 'image_cnn is not defined.'
    
    args.arch = arch
    return args

  def _task_setup(self):
    args = [('batch_size', 'bs4'), ('noise', 'N1en1'), ('step_size', '128'),
    ('base_resolution', 'br1x0'), ('num_samples', 'ns10'), ('extent_samples', 'es5e2'), 
    ('dataset', 'sbpd'), ('mapper_noise', 'mapN0'), ('graph_type', 'ogrid'), ('flip', 'flip0')]
    return utils.DefArgs(args)

  def _task_hook(self, task_vars_args, task_str, args):
    task_vars = task_vars_args.process_string(task_str)
    logging.error('task_vars: %s', task_vars)
    
    noise = str_to_float(task_vars.noise[1:])
    batch_size = int(task_vars.batch_size[2:])
    step_size = int(task_vars.step_size)
    base_resolution = str_to_float(task_vars.base_resolution[2:])
    num_samples = int(task_vars.num_samples[2:])
    extent_samples = str_to_float(task_vars.extent_samples[2:])
    mapper_noise = str_to_float(task_vars.mapper_noise[4:])
    add_flips = int(task_vars.flip[4:]) > 0
    graph_type = task_vars.graph_type

    road_dilate_disk_size = 0

    fovs = [50]
    vs = [0.25 / base_resolution]
    
    top_view_task_params = tle.get_top_view_discrete_env_task_params(
      prob_random=noise, fovs=fovs, view_scales=vs, batch_size=batch_size,
      ignore_roads=True, output_roads=True,
      road_dilate_disk_size=road_dilate_disk_size, map_max_size=None,
      base_resolution=base_resolution, step_size=step_size, top_view=False,
      perturb_views=False, graph_type=graph_type)
    
    mapper_task_params = me.get_mapper_task_params(batch_size=batch_size, 
      num_samples=num_samples, extent_samples=extent_samples, add_flips=add_flips, 
      mapper_noise=mapper_noise, repeat_map_samples=1, test_f=None)
    
    assert(task_vars.dataset in ['campus', 'sbpd'])
    if task_vars.dataset == 'campus':
      env_class = tle.TopViewDiscreteEnv
      camera_param = None

    elif task_vars.dataset == 'sbpd':
      env_class = mp_env.MPDiscreteEnv
      camera_param = utils.Foo(width=256, height=256, z_near=0.05, z_far=20.0,
        fov_horizontal=60., fov_vertical=60., modalities=['rgb'],
        img_channels=3, im_resize=224/256.)
      top_view_task_params.ignore_roads = False
 
    task = utils.Foo()
    task.camera_param = camera_param
    task.seed = 0
    task.dataset_name = task_vars.dataset
    task.num_actions = 4
    
    task.env_class = env_class
    task.env_task_params = top_view_task_params
    task.env_class_2 = me.MapperEnv
    task.env_task_params_2 = mapper_task_params 
    
    args.task = task
    args.env_multiplexer = tle.EnvMultiplexer
    return args 
  
  def compile(self):
    names = ['task', 'arch', 'solver', 'mode']
    for n in names:
      self.def_args += [getattr(self, '_'+n+'_setup')()]
      self.name_args += [n]

  def post_hooks(self, args):
    # We need to adjust parameters such that the mapper returns more images
    # than what it was originally configured to return. We also need to adjust
    # the test_f so that testing is performed on the same number of locations
    # irrespectiev of the number of images returns by the mapper.
    if args.control.max_for_test > 0:
      num_test = int(round(args.arch.test_f * args.task.env_task_params_2.num_samples))
      num_samples = num_test + args.control.max_for_test
      test_f = num_test*1./num_samples
      logging.error('Changing test_f to: %f', test_f)
      logging.error('Changing num_samples to: %d', num_samples)
      args.arch.test_f = test_f
      args.task.env_task_params_2.num_samples = num_samples
    
    # This was added to make sure mapper returns the same samples for when we
    # were trying to generate a heat map of locations.
    args.task.env_task_params_2.test_f = args.arch.test_f
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
