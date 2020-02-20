from _logging import logging
import torch
import numpy as np
from src import utils
from multiprocessing import Process, Queue
import threading
from six.moves.queue import Queue as LocalQueue
import random
from cfgs.config_map_follower import ConfigManager
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_code.model_utils import Conditional_Net
from pytorch_code.train_utils import Data_Bank, SS_Explore
import cv2
from absl import flags, app
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

FLAGS = flags.FLAGS
flags.DEFINE_string('basepath','','base directory for the experiment')
flags.DEFINE_string('expt_name','','expt folder name')
flags.DEFINE_string('run_name','','expt folder name')
flags.DEFINE_string('test_type','','expt folder name')
flags.DEFINE_string('color','','color')
flags.DEFINE_string('label','','')



def get_coords(it):
      iy = it//2; ix = it%2
      return ix, iy

def setup_plots(ax, num_ops, mid_, sz, full_view):
  ax.imshow(0.8*(1-full_view.astype(np.float32)/255.), vmin=-0.5, vmax=1.5, cmap='Greys', origin='lower')
  if mid_ is not None and sz is not None:
      ax.xlim([mid_[0]-sz, mid_[0]+sz])
      ax.ylim([mid_[1]-sz, mid_[1]+sz])
  ax.xticks([])
  ax.yticks([])

def find_sizes(states):
  all_locs = states[0][0,:,::-1]
  for itr in range(len(states)):
    for i in range(len(states[itr])): 
        all_locs = np.concatenate([all_locs,states[itr][i,:,::-1]], axis = 0)
  
  all_locs=all_locs[:,1:]
  min_ = np.min(all_locs, axis=0)
  max_ = np.max(all_locs, axis=0)
  mid_ = (min_+max_)/2.
  sz = np.maximum(1.1*np.max(max_-min_)/2., 100)
  return mid_, sz

def plot_traj(ax, k, c, alpha, lw, ms, label):
    ax.plot(k[:,1], k[:,0], c, alpha=alpha, lw=lw, ms=ms, label=label)

def plot_stitched(ax, states, output_dir, ep_num, line_width, label):
  ax = plt.gca()
  fig = plt.gcf()
  extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).get_points()
  extent[0,1] += 1.0
  #extent[1,0] -= 1.0
  extent[0,0] += 0.25
  extent[1,0] -= 0.25
  extent = matplotlib.transforms.Bbox(extent)
  colr = 'r-'#FLAGS.color
  for k in states[0][0:20]:
      plot_traj(ax, k, colr, 0.4, line_width, line_width, label)
  plt.legend(label, prop={'size': 16}, loc = 1)
  plt.savefig(os.path.join(output_dir, '{0:03d}.pdf'.format(ep_num)),bbox_inches=extent)

def get_top_view_plot(output_dir, states, label, full_view, j, ep_num, with_teacher=True):
  plt.tight_layout()
  #mid_, sz = find_sizes(states)
  setup_plots(plt, len(states), None, None, full_view)
  plot_stitched(plt, states, output_dir, ep_num, 4, label)
  #plt.close()


def load_full_view():
  test_type = FLAGS.test_type
  full_path_name = 'output/active_eval_fwd_models/'
  full_view_file = os.path.join(full_path_name, 'full_view_'+test_type+'.png')
  full_view = cv2.imread(full_view_file)[:,:,0]#.transpose()
  return full_view

def generate_dir_names():
  input_dir = os.path.join(FLAGS.basepath, FLAGS.expt_name, FLAGS.test_type,'')
  output_dir = os.path.join(FLAGS.basepath, FLAGS.expt_name, FLAGS.test_type, 'viz_coverage', FLAGS.run_name)
  if not os.path.exists(output_dir): os.makedirs(output_dir)
  dir_names = [os.path.join(input_dir, FLAGS.run_name + '.pkl')]
  return output_dir, dir_names, [FLAGS.label]

def load_tts(dir_names):
  tts = []
  for dir_name in dir_names:
    file_name = dir_name
    tt = utils.load_variables(file_name)
    tt['states'] = tt['states'][:,:,:,0,:]
    tts.append(tt)
  return tts

def _gen_save_plot(output_dir, tts, label_list):
  full_view = load_full_view()
  for ep_num in range(len(tts[0]['states'])):
    loc_tt = []
    for itr1 in range(len(tts)):
        loc_tt.append(tts[itr1]['states'][ep_num])
    get_top_view_plot(output_dir,loc_tt, label_list, full_view, 0, ep_num)

def worker():
    output_dir, dir_names, label_list = generate_dir_names()
    tts = load_tts(dir_names)
    _gen_save_plot(output_dir, tts, label_list)

def main(_):
    worker()

if __name__ == '__main__':
    app.run(main)
            
