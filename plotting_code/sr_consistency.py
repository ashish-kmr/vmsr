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
flags.DEFINE_string('num_ops','','expt folder name')
flags.DEFINE_string('xname', None, 'x axis info')
flags.DEFINE_string('yname', None, 'x axis info')
flags.DEFINE_string('xtype', None, 'x axis info')
flags.DEFINE_string('legend', 'yes', 'x axis info')
flags.DEFINE_string('eps', 'all', 'x axis info')
flags.DEFINE_string('suffix', '', 'x axis info')



def get_coords(it):
      iy = it//2; ix = it%2
      return ix, iy

def setup_plots(ax, num_ops, mid_, sz, full_view):
  for i in range(num_ops):
      ix, iy = get_coords(i)
      ax[ix,iy].imshow(0.8*(1-full_view.astype(np.float32)/255.), vmin=-0.5, vmax=1.5, cmap='Greys', origin='lower')
      ax[ix,iy].set_xlim([mid_[0]-sz, mid_[0]+sz])
      ax[ix,iy].set_ylim([mid_[1]-sz, mid_[1]+sz])
      ax[ix,iy].get_xaxis().set_ticks([])
      ax[ix,iy].get_yaxis().set_ticks([])
      ax[ix,iy].set_axis_off()

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

def clean_traj(k):
    xyt = [k[0]]
    for i in k:
        if (i[0:2] == xyt[-1][0:2]).all():
            xyt[-1] = i
        else:
            xyt.append(i)
    return np.array(xyt)

def plot_traj(ax, k, c, alpha, lw, ms, label):
    xyt = clean_traj(k)
    xyt = xyt[0:xyt.shape[0]:6,:]
    U = 0.01*np.sin(xyt[:,2]); V = 0.01*np.cos(xyt[:,2])
    ax.plot(k[:,1], k[:,0], c, alpha=alpha, lw=lw, ms=ms, label=label, zorder = 1)
    ax.plot(k[-1,1], k[-1,0], 'ko', lw=lw, ms=ms, label=label, zorder = 1)
    #Q = ax.quiver(xyt[:,1], xyt[:,0], U, V,  units='width', width = 0.008, scale = 0.18, alpha = 0.7, zorder = 1)

def plot_start(ax, xyt,c):
    U = 0.01*np.sin(xyt[:,2]); V = 0.01*np.cos(xyt[:,2])
    Q = ax.quiver(xyt[:,1], xyt[:,0], U, V,  color = c, edgecolor = 'k', \
            units='width', width = 0.008, scale = 0.2, zorder = 4)
    ax.plot(xyt[:,1], xyt[:,0] + 8,'wo',ms = 25, alpha = 0.85)

def plot_combined(ax, states, output_dir, ep_num, color_lst, line_width, label, fig):
  utils.mkdir_if_missing(os.path.join(output_dir, 'combined'))
  utils.mkdir_if_missing(os.path.join(output_dir, 'individual'))
  for n, state_i in enumerate(states):
      ix, iy = get_coords(n)
      for k in state_i[0:20]:
          plot_traj(ax[ix, iy], k, color_lst[n], 0.4, line_width[n], line_width[n], 'subroutine {0:01d}'.format(n))
          if FLAGS.legend == 'yes':
            ax[ix, iy].legend(['subroutine {0:01d}'.format(n)], prop={'size': 30, 'weight':500}, loc = 1)
      #ax[ix,iy].plot(states[0][0,0,1],states[0][0,0,0],'k*',ms=10, zorder = 3)
      plot_start(ax[ix,iy], states[0][0:1,0,:], c = 'k')
      extent = ax[ix,iy].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
      #import pdb; pdb.set_trace()
      bbox_extent = extent.get_points()
      if n==0:
          bbox_extent[0,0]+=0.7; bbox_extent[1,0]-=0.7
      extent = matplotlib.transforms.Bbox(bbox_extent)
      plt.savefig(os.path.join(output_dir, 'individual', '{0:01d}_{1:03d}.pdf'.format(n, ep_num)), bbox_inches=extent)

  plt.savefig(os.path.join(output_dir, 'combined', '{0:03d}.png'.format(ep_num)),bbox_inches='tight')
  #ax.clear()

def get_top_view_plot(output_dir, states, label, full_view, j, ep_num, with_teacher=True):
  num_subplots = max(2,len(states))
  line_width=[4 for i in range(num_subplots)]
  color_lst=['g-','b-','r-','m-', 'k-', 'c-', 'g-', 'b-', 'r-', 'k-']
  plt.tight_layout()
  fig, ax = plt.subplots(2, (num_subplots+1)//2, figsize=(8 * (num_subplots+1)//2,16))
  frame_list=[]
  student_states = states
  mid_, sz = find_sizes(states)
  setup_plots(ax, len(states), mid_, sz, full_view)
  plot_combined(ax, states, output_dir, ep_num, color_lst, line_width, label, fig)
  plt.close(fig)


def load_full_view():
  test_type = FLAGS.test_type
  full_path_name = 'output/active_eval_fwd_models/'
  full_view_file = os.path.join(full_path_name, 'full_view_'+test_type+'.png')
  full_view = cv2.imread(full_view_file)[:,:,0]
  return full_view

def generate_dir_names():
  input_dir = os.path.join(FLAGS.basepath, FLAGS.expt_name, FLAGS.test_type, 'unroll_ops','')
  output_dir = os.path.join(FLAGS.basepath, FLAGS.expt_name, FLAGS.test_type, \
          'viz_unroll_ops', FLAGS.run_name, FLAGS.legend+'_'+FLAGS.suffix)
  if not os.path.exists(output_dir): os.makedirs(output_dir)
  dir_names = [input_dir + FLAGS.run_name + '_{0:01d}_80_.pkl'.format(i) for i in range(int(FLAGS.num_ops))]
  label_list=['sr_{0:02d}'.format(i) for i in range(int(FLAGS.num_ops))]
  return output_dir, dir_names, label_list

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
  if FLAGS.eps == 'all':
      eprange = range(len(tts[0]['states']))
  else:
      eprange = FLAGS.eps.split(',')
      eprange = [int(itr_e) for itr_e in eprange]

  for ep_num in eprange: 
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
            
