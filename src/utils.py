# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Generaly Utilities.
"""
from __future__ import print_function
import six, sys
from six.moves import cPickle
import numpy as np, os, time
from src import file_utils as fu
import logging, hashlib
from contextlib import contextmanager

def get_hashes(names, values):
  hs = []
  for n, v in zip(names, values):
    hs.append('{:s}-{:s}'.format(n, get_hash(v)))
  return hs

def get_hash(v):
  return hashlib.md5(str(v)).hexdigest()

def get_time_str():
  return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def get_rng(rng):
  """Seed of a new rng from the first sample from this rng."""
  return np.random.RandomState(rng.randint(np.iinfo(np.uint32).max))

def copy_rng(rng):
  """Creates a copy of the rng."""
  rng_copy = np.random.RandomState(0)
  rng_copy.set_state(rng.get_state())
  return rng_copy

def test_copy_rng():
  rng = np.random.RandomState(0)
  _ = rng.rand(); r0 = rng.rand();
  
  rng = np.random.RandomState(0)
  _ = rng.rand()

  rng_copy = copy_rng(rng)
  r1 = rng.rand()
  r2 = rng_copy.rand()
  assert(r1 == r2)
  assert(r0 == r1)

def half_life(k):
  return k * np.log(2.*k)

class DefArgs():
  def __init__(self, keys_vals):
    keys = [_[0] for _ in keys_vals]
    default_vals = [_[1] for _ in keys_vals]
    self.keys = keys
    self.default_vals = default_vals
    assert(len(self.keys) == len(self.default_vals))
    for k, v in zip(self.keys, self.default_vals):
      assert(type(k) == str)
      assert(type(v) == str)
  
  def process_string(self, string):
    vals = string.split('_')
    assert(len(vals) <= len(self.keys))
    for i, dv in enumerate(self.default_vals):
      if len(vals) <= i:
        vals.append(dv)
      elif vals[i] == '':
        vals[i] = dv
    vars = Foo()
    for k, v in zip(self.keys, vals):
      setattr(vars, k, v)
    return vars

  def get_default_string(self):
    return '_'.join(self.default_vals)

def test_def_args_string():
  t = [('noise', 'a'), ('batch_size', 'b'), ('flips', 'c')]
  d = DefArgs(t)
  assert(d.get_default_string() == 'a_b_c')

def test_def_args():
  t = [('noise', 'a'), ('batch_size', 'b'), ('flips', 'c')]
  d = DefArgs(t) 
  for t in ['', '__', '_']:
    var = d.process_string(t)
    assert(var.noise == 'a')
    assert(var.batch_size == 'b')
    assert(var.flips == 'c')

  var = d.process_string('aa_b')
  assert(var.noise == 'aa')
  assert(var.batch_size == 'b')
  assert(var.flips == 'c')

def test_timer_record():
  t = Timer()
  with t.record():
    time.sleep(.01)
  t.display(log_at=1, log_str='test timer: ', type='time', mul=1)

class Timer():
  def __init__(self, skip=0, stream='info'):
    self.calls = 0.
    self.start_time = 0.
    self.time_per_call = 0.
    self.time_ewma = 0.
    self.total_time = 0.
    self.last_log_time = 0.
    self.skip = skip
    self.stream = stream

  def tic(self):
    self.start_time = time.time()
  
  def display(self, average=True, log_at=-1, log_str='', type='calls', mul=1, 
    current_time=None):
    if current_time is None: 
      current_time = time.time()
    if self.skip == 0:
      ewma = self.time_ewma * mul / np.maximum(0.01, (1.-0.99**self.calls))
      if type == 'calls' and log_at > 0 and np.mod(self.calls/mul, log_at) == 0:
        _ = []
        getattr(logging, self.stream)('%s: %f seconds / call, %d calls.', log_str, ewma, self.calls/mul)
      elif type == 'time' and log_at > 0 and current_time - self.last_log_time >= log_at:
        _ = []
        getattr(logging, self.stream)('%s: %f seconds / call, %d calls.', log_str, ewma, self.calls/mul)
        self.last_log_time = current_time
    # return self.time_per_call*mul
    return ewma

  def toc(self, average=True, log_at=-1, log_str='', type='calls', mul=1):
    if self.skip > 0:
      self.skip = self.skip-1
    else:
      if self.start_time == 0:
        logging.error('Timer not started by calling tic().')
      t = time.time()
      diff = time.time() - self.start_time
      self.total_time += diff; self.calls += 1.;
      self.time_per_call = self.total_time/self.calls
      alpha = 0.99
      self.time_ewma = self.time_ewma*alpha + (1-alpha)*diff
      self.display(average, log_at, log_str, type, mul, current_time=time)
    
    if average:
      return self.time_per_call*mul
    else:
      return diff
  
  @contextmanager
  def record(self):
    self.tic()
    yield
    self.toc()

class Foo(object):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
  def __str__(self):
    str_ = ''
    for v in vars(self).keys():
      a = getattr(self, v)
      if True: #isinstance(v, object):
        str__ = str(a)
        str__ = str__.replace('\n', '\n  ')
      else:
        str__ = str(a)
      str_ += '{:s}: {:s}'.format(v, str__)
      str_ += '\n'
    return str_

def dict_equal(dict1, dict2):
  assert(set(dict1.keys()) == set(dict2.keys())), "Sets of keys between 2 dictionaries are different."
  for k in dict1.keys():
    assert(type(dict1[k]) == type(dict2[k])), "Type of key '{:s}' if different.".format(k)
    if type(dict1[k]) == np.ndarray:
      assert(dict1[k].dtype == dict2[k].dtype), "Numpy Type of key '{:s}' if different.".format(k)
      assert(np.allclose(dict1[k], dict2[k])), "Value for key '{:s}' do not match.".format(k)
    else:
      assert(dict1[k] == dict2[k]), "Value for key '{:s}' do not match.".format(k)
  return True

def subplot(plt, Y_X, sz_y_sz_x = (10, 10)):
  logging.error('subplot has been deprecated in favor of subplot2. subplot2 additionally outputs axes as a list.')
  Y,X = Y_X
  sz_y, sz_x = sz_y_sz_x
  plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
  fig, axes = plt.subplots(Y, X)
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  return fig, axes

def subplot2(plt, Y_X, sz_y_sz_x=(10,10), space_y_x=(0.1,0.1), T=False):
  Y,X = Y_X
  sz_y, sz_x = sz_y_sz_x
  hspace, wspace = space_y_x
  plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
  fig, axes = plt.subplots(Y, X, squeeze=False)
  plt.subplots_adjust(wspace=wspace, hspace=hspace)
  if T:
    axes_list = axes.T.ravel()[::-1].tolist()
  else:
    axes_list = axes.ravel()[::-1].tolist()
  return fig, axes, axes_list

class TicTocPrint():
  def __init__(self, interval):
    self.interval = interval
    self.last_time = 0
  def log(self, *args):
    t = time.time()
    if t - self.last_time > self.interval:
      logging.error(*args)
      self.last_time = t

def test_tic_toc_print():
  tt = TicTocPrint(0.5)
  tt.log('%d', 0)
  tt.log('%d', 1)
  time.sleep(1)
  tt.log('%d', 2)

def tic_toc_print(interval, string):
  global tic_toc_print_time_old
  if 'tic_toc_print_time_old' not in globals():
    tic_toc_print_time_old = time.time()
    print(string)
  else:
    new_time = time.time()
    if new_time - tic_toc_print_time_old > interval:
      tic_toc_print_time_old = new_time;
      print(string)

def mkdir_if_missing(output_dir):
  if not fu.exists(output_dir):
    try:
      fu.makedirs(output_dir)
    except:
      logging.error("Something went wrong in mkdir_if_missing. "
        "Probably some other process created the directory already.")

def save_variables(pickle_file_name, var, info, overwrite = False):
  if fu.exists(pickle_file_name) and overwrite == False:
    raise Exception('{:s} exists and over write is false.'.format(pickle_file_name))
  # Construct the dictionary
  assert(type(var) == list); assert(type(info) == list);
  for t in info: assert(type(t) == str), 'variable names are not strings'
  d = {}
  for i in range(len(var)):
    d[info[i]] = var[i]
  with fu.fopen(pickle_file_name, 'wb') as f:
    cPickle.dump(d, f, cPickle.HIGHEST_PROTOCOL)

def load_variables(pickle_file_name):
  if fu.exists(pickle_file_name):
    if sys.version_info >= (3, 0):
      with fu.fopen(pickle_file_name, 'rb') as f:
        d = cPickle.load(f, encoding='latin1')
    else:
      with fu.fopen(pickle_file_name, 'r') as f:
        d = cPickle.load(f)
    return d
  else:
    raise Exception('{:s} does not exists.'.format(pickle_file_name))

def voc_ap_fast(rec, prec):
  rec = rec.reshape((-1,1))
  prec = prec.reshape((-1,1))
  z = np.zeros((1,1)) 
  o = np.ones((1,1))
  mrec = np.vstack((z, rec, o))
  mpre = np.vstack((z, prec, z))
  mpre_ = np.maximum.accumulate(mpre[::-1])[::-1]
  I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
  ap = np.sum((mrec[I] - mrec[I-1]) * mpre[I])
  return np.array(ap).reshape(1,)

def voc_ap(rec, prec):
  rec = rec.reshape((-1,1))
  prec = prec.reshape((-1,1))
  z = np.zeros((1,1)) 
  o = np.ones((1,1))
  mrec = np.vstack((z, rec, o))
  mpre = np.vstack((z, prec, z))
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])

  I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
  ap = 0;
  for i in I:
    ap = ap + (mrec[i] - mrec[i-1])*mpre[i];
  return ap

def tight_imshow_figure(plt, figsize=None):
  fig = plt.figure(figsize=figsize)
  ax = plt.Axes(fig, [0,0,1,1])
  ax.set_axis_off()
  fig.add_axes(ax)
  return fig, ax

def gridspec_subplot(plt, figsize=(9,6), gridsize=(2,3)):
  """How to use GridSpec. 9in wide, 6in tall, 2 rows and 3 columns (life is not
  easy)"""
  fig = plt.figure(figsize=figsize)
  gs = gridspec.GridSpec(gridsize)
  return fig, gs

def get_538_cm():
  tt =  ['008fd5', 'fc4f30', 'e5ae38', '6d904f', '8b8b8b', '810f7c']
  tt = ['#'+x for x in tt]
  return tt

def calc_pr(gt, out, wt=None, fast=False):
  """Computes VOC 12 style AP (dense sampling).
  returns ap, rec, prec"""
  if wt is None:
    wt = np.ones((gt.size,1))

  gt = gt.astype(np.float64).reshape((-1,1))
  wt = wt.astype(np.float64).reshape((-1,1))
  out = out.astype(np.float64).reshape((-1,1))

  gt = gt*wt
  tog = np.concatenate([gt, wt, out], axis=1)*1.
  ind = np.argsort(tog[:,2], axis=0)[::-1]
  tog = tog[ind,:]
  cumsumsortgt = np.cumsum(tog[:,0])
  cumsumsortwt = np.cumsum(tog[:,1])
  prec = cumsumsortgt / cumsumsortwt
  rec = cumsumsortgt / np.sum(tog[:,0])

  if fast:
    ap = voc_ap_fast(rec, prec)
  else:
    ap = voc_ap(rec, prec)
  return ap, rec, prec

def str_to_float(x):
  return float(x.replace('x', '.').replace('n', '-'))
