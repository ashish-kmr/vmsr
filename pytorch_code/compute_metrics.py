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
#from pytorch_code.model_utils import Net, Conditional_Net
from pytorch_code.train_utils import Data_Bank, SS_Explore
import skfmm
import numpy.ma as ma
import cv2
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import seaborn as sns
from absl import flags, app
import json

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('logdir_prefix',
                'output/mp3d/operators_invmodel_lstm_models_sn5/',
                    'base directory for the experiment')
    flags.DEFINE_string('expt_name', None, 'args for experiment')
    flags.DEFINE_string('test_env', 'area3', 'env to test on')
    flags.DEFINE_string('method_str', '', '')
    flags.DEFINE_string('standard_str', '', '')
    flags.DEFINE_string('label', '', '')



def getBC(succ):
    mu = np.mean(succ)
    np.random.seed(1)
    replicates = []
    for i in range(1000):
        boot = np.random.choice(succ,size=len(succ),replace=True)
        replicates.append(np.mean(boot))

    ci = _bciPercentileBias(np.array(replicates),mu,0.05)
    return np.array([ci[0],mu,ci[1]])

def _pctclip(X,p):
    """np.percentile but clipping to [0,100]"""
    p = [min(100,max(0,v)) for v in p]
    return np.percentile(X,p)

def _bciPercentileBias(replicates,allData,alpha):
    """Simple percentile CI with bias correction"""
    #The jackknife, the bootstrap and other resampling plans 
    #(See page 118)
    #https://statistics.stanford.edu/sites/default/files/BIO%2063.pdf
    alpha = alpha/2

    #get z for the bias and the desired alpha
    #note that ppf(0.5) = 0 so if the correction is positive, this pushes the 
    #distribution to the right; you can think of this as reintepreting the 
    #samples on either side of the median: ideally you should have 50% on 
    #either side

    z0 = stats.norm.ppf(np.mean(replicates <= allData))
    #this will be ~1.96 for a 95% CI
    za = stats.norm.ppf(alpha)

    higha, lowa = stats.norm.cdf(2*z0-za), stats.norm.cdf(2*z0+za)
    return _pctclip(replicates,[lowa*100,higha*100])


def compute_norm_coverage(states, e):
    # rollouts x path length
    states = states[:,:,0:2].astype(np.int32)
    #import pdb; pdb.set_trace()
    _, dd = _compute_obst_distance_field(e.task.env.exp_nav, states)
    _, dd_base = _compute_obst_distance_field(e.task.env.exp_nav, states[:,0:1])
    rel_idx = np.where(dd_base <= 4*states.shape[1])
    return np.mean(dd[rel_idx[0], rel_idx[1]])
    #return dst_f

def compute_coverage(states, e):
    # rollouts x path length
    states = states[:,:,0:2].astype(np.int32)
    #import pdb; pdb.set_trace()
    dst_f, _ = _compute_obst_distance_field(e.task.env.exp_nav, states)
    return dst_f
  
def _compute_obst_distance_field(mp_env, states):
    t = mp_env.map['traversible_cc']
    #print('masked',t.shape)
    idx_lst = np.argwhere(t)
    masked_t = ma.masked_values(t*1, 0)
    idx_x = states[:,:,0].reshape([1,-1])
    idx_y = states[:,:,1].reshape([1,-1])
    masked_t[idx_x,idx_y]=0
    dd = skfmm.distance(masked_t, dx=1)
    #dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
    #import pdb; pdb.set_trace()
    dd = ma.filled(dd, np.max(dd)+1)
    return np.mean(dd[idx_lst[:,0],idx_lst[:,1]]), dd
    #return dd

def _compute_collisions(traj, act):
    coll_d = []
    for i in range(act.shape[0]):
        dist_i = np.sum(np.abs(traj[i] - traj[i+1])) 
        if act[i] ==3 and dist_i == 0:
            coll_d.append(4)
        else:
            coll_d.append(act[i,0])
    return coll_d
       
def _compute_collisions_batch(traj, act):
    coll_list = []
    for i,j in zip(traj,act):
        coll_list += _compute_collisions(i,j)
    return coll_list

def _max_dist(traj, e):
    _, dd = _compute_obst_distance_field(e.task.env.exp_nav, traj[0:1,0:1,0:2].astype(np.int32))
    reshaped_states = traj.reshape([-1,3])
    val = dd[reshaped_states[:,0].astype(np.int32), reshaped_states[:,1].astype(np.int32)]
    return np.max(val)

def plot_curves(save_dir, xaxlst, methods, labels, xlab, ylab, xlim=None, ylim=None, bc_ = True):
    sns.set()
    sns.set_context("talk")

    plt.figure(figsize=(6,6), dpi=100)

    if bc_:
        for indeps, results, lab in zip(xaxlst, methods, labels):
            plt.plot(indeps,results[:,1],label=lab)
            plt.fill_between(indeps,results[:,0],results[:,2],alpha=0.2,antialiased=True)
    else:
        for indeps, results, lab in zip(xaxlst, methods, labels):
            plt.plot(indeps,results,label=lab)


    plt.ylabel(xlab)
    plt.xlabel(ylab)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()
    plt.show()
    plt.savefig(save_dir)

def plot_dist(save_dir, method, label, xlab, ylab, xlim=None, ylim=None, bins = 'auto', histtype = 'bar', density = True):
    sns.set()
    sns.set_context("talk")

    plt.figure(figsize=(6,6), dpi=100)

    plt.hist(method,label=label, density=density, bins = bins, rwidth = 0.5, align = 'left', histtype = histtype)

    plt.ylabel(xlab)
    plt.xlabel(ylab)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()
    plt.show()
    plt.savefig(save_dir)

def plot_coll_dist(save_dir, coll, labels):
    for n,i in enumerate(coll):
        plot_dist(save_dir + '/coll'+labels[n]+'.pdf', i, labels[n], 'Action', '', bins = [0,1,2,3,4,5], ylim = (0,1))

def plot_max_dist(save_dir, max_dist, labels):
    methods = []
    xax = []
    xlim = 0
    for i in max_dist:
        methods.append(np.sort(np.array(i)*5.0/100.0))
        xax.append((np.array(range(len(i), 0, -1)))/(1.0*len(i)))
        xlim = max(xlim, np.max(np.array(i)*5.0/100.0))

    plot_curves(save_dir + '/cumulative_maxdist.pdf', methods, xax, labels, 'Fraction of Rollouts', 'Max Dist (in m)', ylim = (0,1.02),xlim = (0,xlim), bc_ = False)

def plot_coverage(save_dir, cov, labels):
    methods = []
    indeps = [range(cov[0].shape[0]) for i in cov]
    ylim = 0
    for i in cov:
        bs_estimate = []
        for j in range(i.shape[0]):
            bs_estimate.append(getBC(np.array(i[j])*5.0/100.0))
        methods.append(np.array(bs_estimate))
        ylim = max(np.max(np.array(i) * 5.0 / 100.0), ylim)
    plot_curves(save_dir + '/cumulative_coverage.pdf', indeps, methods, labels, 'MDT (in m)', 'Episodes', ylim = (0,ylim))

def load_pkls(dir_names):
    tts_l = []
    for dir_name in dir_names:
        file_name = dir_name
        tt = utils.load_variables(file_name)
        tts_l.append(tt)
    return tts_l

def process_tts(tts_l):
    for i in range(len(tts_l)):
        tts_l[i]['states'] = tts_l[i]['states'][:,:,:,0,:]
        if len(tts_l[i]['gt_action'].shape)<4:
            tts_l[i]['gt_action'] = np.expand_dims(tts_l[i]['gt_action'],3)
    return tts_l

def concatenate_states(stl):
  cum_st_list = (stl[0])
  for i in range(1,len(stl)):
      cum_st_list = np.concatenate((cum_st_list, stl[i]), axis = 0)
  return cum_st_list

def add_data(mdt_len, ftype):
  pth = 'metric_eval/' + ftype + '/' + FLAGS.standard_str + '.json'
  if not os.path.exists(pth):
      f=open(pth, 'w')
      json.dump({}, f)
      f.close()

  f = open(pth)
  data = json.load(f)
  data[FLAGS.label] = mdt_len
  f=open(pth, 'w')
  json.dump(data, f)
  f.close()

def ablate_trajlen(tts_base,e):
  mdt_len = []
  max_dist_len = []
  import copy
  traj_len = tts_base[0]['states'].shape[2]
  #import pdb; pdb.set_trace()
  for oitr in range(50,traj_len,50):
      tt_orig = copy.deepcopy(tts_base)
      for itr in range(len(tt_orig)): 
          tt_orig[itr]['states'] = tt_orig[itr]['states'][:,:,0:oitr]
          tt_orig[itr]['gt_action'] = tt_orig[itr]['gt_action'][:,:,0:oitr-1]
      for n, method in enumerate(tt_orig): 
          metrics_I = (compute_metrics_i(method, e))
          mdt_len.append(metrics_I['new_mdt_list'])
          max_dist_len.append(metrics_I['max_dist_list'])
  mdt_len = [itr for itr in mdt_len]
  max_dist_len = [itr for itr in max_dist_len]
  add_data(mdt_len, 'mdt')
  add_data(max_dist_len, 'max_dist')
  print(mdt_len)
  print(max_dist_len)
  #import pdb; pdb.set_trace()

def concatenate_states(stl, n=100):
  assert(len(stl) >= n)
  cum_st_list = np.concatenate(stl[:n], axis=0)
  return cum_st_list

def compute_new_coverage(states, e):
    cov_list = []
    for i in range(states.shape[0]):
        concat_st = np.array(np.expand_dims(np.concatenate(states[i]),0))
        cov_list.append(compute_coverage(concat_st, e)) 

    return cov_list


def compute_metrics_i(tt, e, n=100):
    coll_list = []; max_dist = []; coll_frac_list = [] 
    print(tt['states'].shape)
    for ep_num in range(len(tt['states'])):
      state_list = tt['states'][ep_num]
      act_list = tt['gt_action'][ep_num]
      coll_list_i = _compute_collisions_batch(state_list, act_list)
      coll_list += coll_list_i 
      coll_list_i = np.array(coll_list_i)
      coll_frac = len(np.where(coll_list_i == 4)[0])*1.0/len(np.where(coll_list_i >= 3)[0])
      coll_frac_list.append(coll_frac)
      max_dist.append(_max_dist(state_list, e))
    
    coll_list = np.array(coll_list)
    cum_st_list = concatenate_states(tt['states'], n=n)
    cov_val = compute_coverage(cum_st_list, e) * 5.0/100.0      
    new_cov_val = np.array(compute_new_coverage(tt['states'], e)) * 5.0/100.0
    max_dist = (np.array(max_dist) * 5.0/100.0).tolist()
    coll_count = len(np.where(coll_list[:] == 4)[0])
    max_dist_val = np.mean(max_dist)
    max_dist_val_median = np.median(max_dist)
    return {'num_trajs': cum_st_list.shape[0], \
            'mdt': cov_val, 'mdt_new': np.mean(new_cov_val), 'max_dist': max_dist_val, \
            'collisions': coll_count, \
            'max_dist_median':max_dist_val_median, \
            'act_dist': np.histogram(tt['gt_action'], bins=[0,1,2,3,4])[0].tolist(), \
            'max_dist_list': max_dist, 'new_mdt_list': new_cov_val.tolist(), \
            'coll_frac_list': coll_frac_list}

def _test_env():
  import sys
  test_type = FLAGS.test_env
  standard_str = FLAGS.standard_str
  #'n{:04d}_inits{:02d}_or{:02d}_unroll{:03d}'.format(\
  #        FLAGS.num_runs, FLAGS.num_inits, FLAGS.num_orients, FLAGS.unroll_length)
  method_str = FLAGS.method_str #'{:010d}_{:03d}_{:03d}'.format(FLAGS.snapshot, 9, 9)
  fname = standard_str + '.' + method_str
  save_dir = os.path.join(FLAGS.logdir_prefix, FLAGS.expt_name, FLAGS.test_env, fname + '_wmd.json')
  #if not os.path.exists(save_dir):
  #    os.makedirs(save_dir)
  rng=np.random.RandomState(19)
  cm = ConfigManager()
  args = cm.process_string('bs1_N2en1_40_8_16_18_1____mp3d_vp0______TN0_forward_demonstartion' + \
          '_____dilate1_multi1.v0_ns40_sn5_frz0_bn1_dr64_one_fsynth_dnc2_gru_demon.dlw1e1_rlw1en1_' + \
          'ent0e0_lr1en4_adam2+bench_'+test_type)
  env = args.env_multiplexer(args.task, 0, 1)
  e,eid = env.sample_env(rng)

  ## Process the inputs
  labels = ['']
  dir_names = [os.path.join(FLAGS.logdir_prefix, FLAGS.expt_name, FLAGS.test_env, fname + '.pkl')]
  tts_base = load_pkls(dir_names)
  tts_base = process_tts(tts_base)
  #coverage_list = []
  '''
  for n, method in enumerate(tts_base):
      metrics = compute_metrics_i(method, e)
      with open(save_dir, 'w') as f:
          json.dump(metrics, f, sort_keys=True, separators=(',', ': '), indent=4)
          #print(FLAGS.expt_name, metrics['max_dist'], metrics['max_dist_median'])
          print(json.dumps(metrics, sort_keys=True, separators=(',', ': '), indent=4))
          #print(labels[n])
          #for k in metrics_i.keys(): print(k,metrics_i[k])
  '''
  ablate_trajlen(tts_base,e)
  #import pdb; pdb.set_trace()
      
  #plot_coverage(save_dir, cumulative_coverage, labels)

def main(_):
  _test_env()

if __name__ == '__main__':
  app.run(main)
                                                                                                                              
