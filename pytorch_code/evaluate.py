from _logging import logging
import torch
import sys
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
from pytorch_code.model_utils import Conditional_Net, Resnet18_c, \
  Conditional_Net_RN, ActionConditionalLSTM, Latent_Dist_RN, Conditional_Net_RN, \
  Latent_Dist_RN, Latent_NetV, Conditional_Net_RN5, Conditional_Net_RN5N, \
  Latent_Dist_RN5, InverseNetEF_RN
from pytorch_code.train_utils import Data_Bank, SS_Explore, FPVImageDataset, TrajectoryDataset, parse_args
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from absl import flags, app
from pytorch_code.compute_metrics import load_pkls, process_tts, compute_metrics_i 
sys.path.insert(0, 'rl/pytorch-a2c-ppo-acktr')
from gym import spaces
import json

FLAGS = flags.FLAGS
flags.DEFINE_string('logdir_prefix',
    'output/mp3d/', 'base directory for the experiment')
flags.DEFINE_string('expt_name', None, 'args for experiment')
flags.DEFINE_integer('snapshot', 0, 'init from')
flags.DEFINE_integer('num_operators', 4, 'number of operators')
flags.DEFINE_string('test_env', 'area3', 'env to test on')
flags.DEFINE_string('model_type', 'our', 'Model to evaluate')
flags.DEFINE_integer('unroll_length', 80, '')
flags.DEFINE_integer('num_inits', 8, '')
flags.DEFINE_integer('num_orients', 5, '')
flags.DEFINE_integer('num_runs', 100, '')
flags.DEFINE_boolean('stable_mdt', False, '')
flags.DEFINE_integer('gpu_id', 0, '')
flags.DEFINE_boolean('rerun', True, '')
flags.DEFINE_integer('randor', 0, '')
flags.DEFINE_string('sampling', 'prob', '')
flags.DEFINE_string('trtype', '', '')
flags.DEFINE_float('p_forward', 0.66, 'bias for forward policy')

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
class CollavModel:
    def __init__(self, rng):
        self.rng = np.random.RandomState(rng.randint(1e6))
        self.batch_size = FLAGS.num_inits
        self.init_rots = []
        rot_jmp = int(np.ceil(12/FLAGS.num_inits))
        for itr in range(FLAGS.num_inits):
            r1 = rot_jmp * itr; r2 = 1
            if r1>6:
                r1 = 12-r1
                r2 = 2
            self.init_rots.append((r2,r1))
    def reset(self):
        self.f_act = [[] for itr in range(self.batch_size)]

    def sample_rotation_act(self, rotval = None):
        rotation_list = []
        if rotval is not None:
            rotation = rotval[0]
            rcount = rotval[1]
        else:
            rotation = 1 + self.rng.randint(2)
            rcount = 1 + self.rng.randint(6)

        rotation_list = [rotation for itr in range(rcount)]
        return rotation_list

    def act(self, img_d, collision_f, new_ep):
        act_to_take = [3 for itr in range(self.batch_size)]
        for itr in range(self.batch_size):
            if new_ep:
                self.f_act[itr] = self.sample_rotation_act(self.init_rots[itr])

            if collision_f[itr]:
                self.f_act[itr] = self.sample_rotation_act()

            if len(self.f_act[itr])>0: 
                act_to_take[itr] = self.f_act[itr][-1]
                self.f_act[itr] = self.f_act[itr][:-1]

        return act_to_take, np.zeros([self.batch_size, 4])

    def get_str(self):
        file_name = 'collav'
        return file_name


class StatModel:
    def __init__(self, rng, p_forward=0.66):
        self.rng = np.random.RandomState(rng.randint(1e6))
        '''
        act_list=[]
        for i in range(2000):
            e,eid = env.sample_env(self.rng)
            e.task.env.task_params.outputs.top_view = False
            init_env_state = e.reset(self.rng)
            input = e.get_common_data()
            input = e.pre_common_data(input)
            act_list.append(input['teacher_actions'])
            if np.mod(i,100) == 0: print(i)
        act_list = np.array(act_list).reshape([1,-1])
        self.aff = np.histogram(act_list,bins=[0,1,2,3,4])[0].astype(np.float)
        self.aff = self.aff/np.sum(self.aff)
        '''
        self.p_forward = p_forward
        p = p_forward
        self.aff = np.array([0., 0.5-p/2, 0.5-p/2, p]) 
        self.batch_size = FLAGS.num_inits
        
    def reset(self):
        pass

    def act(self, img_d, collision_f, new_ep):
        aff = [self.aff for itr in range(self.batch_size)] 
        act_to_take = [self.rng.choice(a=4,p=aff[itr]) for itr in range(self.batch_size)]
        return act_to_take, aff

    def get_str(self):
        file_name = 'stat_{:0.3f}'.format(self.p_forward)
        return file_name

class RandomModel:
    def __init__(self, rng):
        self.rng = np.random.RandomState(rng.randint(1e6))
        self.batch_size = FLAGS.num_inits

    def reset(self):
        pass
    
    def act(self, img_d, collision_f, new_ep):
        aff = [0.25*np.array([1.,1.,1.,1.]) for itr in range(self.batch_size)]
        act_to_take = [self.rng.choice(a=4,p=aff[itr]) for itr in range(self.batch_size)]
        return act_to_take, aff

    def get_str(self):
        file_name = 'random'
        return file_name

class DIAYN(object):
    def __init__(self, snapshot, save_path, device, reset_f, latent_rf,
        feature_dim=512, hidden_lstm_size=256, num_ops=4, sampling = 'prob'):
        self.sampling = sampling
        model_file = os.path.join(save_path, 'models', 'model_{:010d}.pt'.format(snapshot))
        assert(os.path.exists(model_file)), '{:s} does not exist'.format(model_file)
        actor_critic = torch.load(model_file)[0]
        actor_critic.base.image_cnn = actor_critic.base.image_cnn.eval()
        actor_critic = actor_critic.eval()
        actor_critic.to(device)
        actor_critic.base.image_cnn.to(device)
        actor_critic.device = device
        self.actor_critic = actor_critic
        if True:
          model_dir = os.path.join(save_path, 'models-sane', '{:010d}'.format(snapshot))
          utils.mkdir_if_missing(model_dir)
          linear_file = os.path.join(save_path, 'models-sane', '{:010d}'.format(snapshot), 'linear_actor.pth')
          torch.save(self.actor_critic.dist.linear.state_dict(), linear_file)
          cnn_file = os.path.join(save_path, 'models-sane', '{:010d}'.format(snapshot), 'cnn.pth')
          torch.save(self.actor_critic.base.image_cnn.state_dict(), cnn_file)
          if hasattr(self.actor_critic.base, 'gru'):
            gru_file = os.path.join(save_path, 'models-sane', '{:010d}'.format(snapshot), 'gru.pth')
            torch.save(self.actor_critic.base.gru.state_dict(), gru_file)
        
        self.reset_freq = reset_f
        self.num_ops = num_ops
        self.latent_reset_freq = latent_rf
        self.feature_dim = feature_dim
        #TODO(s-gupta): Figure out a programatic wy to do this. hidden_lstm_size
        self.hidden_lstm_size = 512 
        self.device = device
        self.rng_act = np.random.RandomState(0)
        self.rng_latent = np.random.RandomState(0)
        self.batch_size = FLAGS.num_inits
    
    def reset(self):
        self.hidden_state = torch.zeros(self.batch_size, self.hidden_lstm_size).float().to(self.device)
        self.eval_masks = torch.zeros([self.batch_size, 1]).float().to(self.device)
        self.latent_i = torch.zeros(self.batch_size, self.num_ops).float().to(self.device)
        self.act_taken = torch.zeros([self.batch_size, 1, 4]).float().to(self.device)
        self.prev_act = torch.zeros(self.batch_size, 4).float().to(self.device)
        self.step_n = 0

    def act(self, img_d, collision_f, new_ep):
        if np.mod(self.step_n, self.latent_reset_freq) == 0:
            self.latent_i = torch.zeros(self.batch_size, self.num_ops).float().to(self.device)
            sampled_latent_i = [self.rng_latent.choice(self.num_ops) 
              for _ in range(self.latent_i.shape[0])]
            self.latent_i[range(self.batch_size), sampled_latent_i] = 1
        
        if self.reset_freq > 0 and np.mod(self.step_n, self.reset_freq) == 0:
            self.hidden_state = torch.zeros(self.batch_size, self.hidden_lstm_size).float().to(self.device)
            self.eval_masks = torch.zeros(self.batch_size, 1).float().to(self.device)
        
        inputs = {'view': img_d.view(-1,3,224,224), 'latent': self.latent_i, 'prev_act': self.prev_act}
        value, action, action_log_prob, self.hidden_state, outputs = self.actor_critic.act(
          inputs, self.hidden_state, self.eval_masks, deterministic=True, return_probs=True) 
        outputs = outputs.cpu().detach().numpy()
        if self.sampling == 'prob':
            act_to_take = [self.rng_act.choice(4, p=o) for o in outputs]
        elif self.sampling == 'max':
            act_to_take = np.argmax(outputs,1).tolist()
        
        # After the first step, eval_masks should be 1, that is what copies
        # over the state forward.
        self.eval_masks = torch.ones([self.batch_size, 1]).to(self.device)
     
        self.step_n = self.step_n + 1
        # print(torch.mean(img_d.view(img_d.shape[0],-1), 1))
        # print(outputs)
        return act_to_take, outputs

    def get_str(self):
      addstr = ''
      if self.sampling == 'max': addstr = '_max'
      file_name = '{:010d}_{:03d}_{:03d}{}'.format(FLAGS.snapshot, self.reset_freq, self.latent_reset_freq, addstr)
      return file_name


class Curiosity:
    def __init__(self, snapshot, save_path, device, reset_f,
        feature_dim=512, hidden_lstm_size=256, sampling = 'prob'):
        # input_res = 224
        # observation_space = spaces.Dict({
        #   'view': spaces.Box(low=0, high=255, shape=(3, input_res, input_res), dtype=np.float32),
        # })
        # actor_critic = NavBase(observation_space, 'Nav', recurrent, hidden_lstm_size, feature_dim)
        self.sampling = sampling
        model_file = os.path.join(save_path, 'models', 'model_{:010d}.pt'.format(snapshot))
        assert(os.path.exists(model_file)), '{:s} does not exist'.format(model_file)
        actor_critic = torch.load(model_file)[0]
        actor_critic.base.image_cnn = actor_critic.base.image_cnn.eval()
        actor_critic = actor_critic.eval()
        actor_critic.to(device)
        actor_critic.base.image_cnn.to(device)
        actor_critic.device = device
        self.actor_critic = actor_critic
        model_dir = os.path.join(save_path, 'models-sane', '{:010d}'.format(snapshot))
        utils.mkdir_if_missing(model_dir)
        linear_file = os.path.join(save_path, 'models-sane', '{:010d}'.format(snapshot), 'linear_actor.pth')
        torch.save(self.actor_critic.dist.linear.state_dict(), linear_file)
        cnn_file = os.path.join(save_path, 'models-sane', '{:010d}'.format(snapshot), 'cnn.pth')
        torch.save(self.actor_critic.base.image_cnn.state_dict(), cnn_file)
        if hasattr(self.actor_critic.base, 'gru'):
          gru_file = os.path.join(save_path, 'models-sane', '{:010d}'.format(snapshot), 'gru.pth')
          torch.save(self.actor_critic.base.gru.state_dict(), gru_file)
        
        self.reset_freq = reset_f
        self.feature_dim = feature_dim
        self.hidden_lstm_size = 512 #TODO(s-gupta): Figure out a programatic wy to do this. hidden_lstm_size
        self.device = device
        self.rng_act = np.random.RandomState(0)
        self.batch_size = FLAGS.num_inits
    
    def reset(self):
        self.hidden_state = torch.zeros([self.batch_size, self.hidden_lstm_size]).to(self.device)
        self.eval_masks = torch.zeros([self.batch_size, 1]).to(self.device)
        self.step_n = 0

    def act(self, img_d, collision_f, new_ep):
        hidden_state = self.hidden_state
        # We are getting probabilities and then sampling from it by ourselves.
        value, action, action_log_prob, hidden_state, outputs = self.actor_critic.act(
          {'view': img_d.view(-1,3,224,224)}, self.hidden_state, self.eval_masks, 
          deterministic=True, return_probs=True)
        # After the first step, eval_masks should be 1, that is what copies
        # over the state forward.
        eval_masks = torch.ones([self.batch_size, 1]).to(self.device)
        
        if self.reset_freq > 0 and np.mod(self.step_n, self.reset_freq) == 0:
            hidden_state = hidden_state*0.0
            eval_masks = torch.zeros([self.batch_size, 1]).to(self.device)

        outputs = outputs.cpu().detach().numpy()
        if self.sampling == 'prob':
            act_to_take = [self.rng_act.choice(4, p=o) for o in outputs]
        elif self.sampling == 'max':
            act_to_take = np.argmax(outputs,1).tolist()

        self.hidden_state = hidden_state
        self.eval_masks = eval_masks
        self.step_n = self.step_n + 1
        # print(torch.mean(img_d.view(img_d.shape[0],-1), 1))
        # print(outputs)
        return act_to_take, outputs

    def get_str(self):
      addstr = ''
      if self.sampling == 'max': addstr = '_max'
      file_name = '{:010d}_{:03d}_{:03d}{}'.format(FLAGS.snapshot, self.reset_freq, 999, addstr)
      return file_name

class OurModel:
    def __init__(self, snapshot, save_path, device, reset_f, latent_rf, num_ops=4,
        feature_dim=512, hidden_lstm_size=256, use_aff = 'yes', cav = False, sampling = 'prob',
        trtype = ''):
        self.sampling = sampling
        self.trtype = trtype
        self.featurizer=Conditional_Net_RN5N(feat_dim = feature_dim)
        self.act_con_lstm = ActionConditionalLSTM(feature_dim+4+num_ops, hidden_lstm_size, 4, 39)
        self.latent_dist = Latent_Dist_RN5(num_ops)

        self.featurizer.load_state_dict(torch.load(os.path.join(save_path,
          '{0:03d}_featurizer.pth'.format(snapshot))))
        self.act_con_lstm.load_state_dict(torch.load(os.path.join(save_path,
          '{0:03d}'.format(snapshot)+'_lstm.pth')))
        self.latent_dist.load_state_dict(torch.load(os.path.join(save_path,
          '{0:03d}'.format(snapshot)+'_latent_dist{}.pth'.format(trtype))))

        self.featurizer.to(device)
        self.act_con_lstm.to(device)
        self.latent_dist.to(device)

        self.featurizer.eval()
        self.act_con_lstm.eval()
        self.latent_dist.eval()


        self.latent_reset_freq = latent_rf
        self.reset_freq = reset_f

        self.feature_dim = feature_dim
        self.hidden_lstm_size = hidden_lstm_size
        self.num_ops = num_ops
        self.device = device
        self.rng_act = np.random.RandomState(0)
        self.rng_latent = np.random.RandomState(0)
        self.rng = np.random.RandomState(0)
        self.batch_size = FLAGS.num_inits
        self.use_aff = use_aff
        self.cav = cav
    
    def reset(self):
        self.hidden_state = torch.zeros([1, self.batch_size, self.hidden_lstm_size]).to(self.device)
        self.latent_i = torch.tensor(np.zeros([self.batch_size,1,self.num_ops])).float().to(self.device)
        self.act_taken = torch.zeros([self.batch_size, 1, 4])
        self.step_n = 0
        self.f_act = [[] for itr in range(self.batch_size)]

    def sample_rotation_act(self):
        rotation_list = []
        rotation = 1 + self.rng.randint(2)
        rcount = 1 + self.rng.randint(6)
        rotation_list = [rotation for itr in range(rcount)]
        return rotation_list

    def act(self, img_d, collision_f, new_ep):
        sz0, sz1 = img_d.shape[0:2]
        latent_i = self.latent_i
        hidden_state = self.hidden_state
        img_d_rs = img_d.reshape([-1,3,224,224])
        if np.mod(self.step_n, self.latent_reset_freq) == 0:
            latent_i=0.0*latent_i
            if self.use_aff == 'yes':
                latent_i_dist = F.softmax(self.latent_dist(img_d_rs), 1).cpu().detach().numpy()
            elif self.use_aff == 'no':
                latent_i_dist = 1.0*np.ones([img_d_rs.shape[0], self.num_ops])/(1.*self.num_ops)
            elif self.use_aff == 'th':
                latent_i_dist = F.softmax(self.latent_dist(img_d_rs), 1).cpu().detach().numpy()
                th_idx = np.where(latent_i_dist > 1.0/self.num_ops)
                latent_i_dist = 0.0*latent_i_dist
                latent_i_dist[th_idx] = 1.0
                latent_i_dist = np.array([dist_i/np.sum(dist_i) for dist_i in latent_i_dist])
            elif self.use_aff[0:3] == 'th_':
                thresh = float(self.use_aff[3:])
                latent_i_dist = F.softmax(self.latent_dist(img_d_rs), 1).cpu().detach().numpy()
                th_idx = np.where(latent_i_dist > thresh)
                latent_i_dist = 0.0*latent_i_dist
                latent_i_dist[th_idx] = 1.0
                latent_i_dist = np.array([dist_i/np.sum(dist_i) for dist_i in latent_i_dist])
            elif self.use_aff == 'afmax':
                latent_i_dist = np.zeros([img_d_rs.shape[0], self.num_ops])
                mx_idx = np.argmax(self.latent_dist(img_d_rs).cpu().detach().numpy(),1)
                latent_i_dist[np.arange(img_d_rs.shape[0]),mx_idx] = 1.0
            #sampled_latent_i = self.rng_latent.choice(len(latent_i_dist), [1], p = latent_i_dist)[0]
            sampled_latent_i = [self.rng_latent.choice(self.num_ops, p = latent_i_dist[litr]) for litr in range(len(latent_i_dist))]
            latent_i[range(self.batch_size),0,sampled_latent_i]=1

        features = self.featurizer(img_d_rs).reshape([sz0, sz1, self.feature_dim])
        concat_features = torch.cat([features, self.act_taken.to(self.device), latent_i], 2)
        if np.mod(self.step_n, self.reset_freq) == 0:
            hidden_state = hidden_state*0.0
        
        outputs, hidden_state = self.act_con_lstm(concat_features, hidden_state)
        outputs = F.softmax(outputs, 1).cpu().detach().numpy()
        if self.cav:
            for itr in range(len(collision_f)): 
                if collision_f[itr]:
                    self.f_act[itr] = self.sample_rotation_act()

        if self.sampling == 'prob':
            act_to_take = [self.rng_act.choice(4, p=outputs[itra]) for itra in range(len(outputs))]
        elif self.sampling == 'max':
            act_to_take = np.argmax(outputs,1).tolist()
        if self.cav:
            for itr in range(len(outputs)):
                if len(self.f_act[itr]) > 0:
                    act_to_take[itr] = self.f_act[itr][-1]
                    self.f_act[itr] = self.f_act[itr][:-1]

        self.hidden_state = hidden_state
        self.latent_i = latent_i
      
        # store what action was taken for next iteration
        self.act_taken = torch.zeros([self.batch_size,1,4])
        self.act_taken[range(self.batch_size),0,act_to_take] = 1.0
        self.step_n = self.step_n + 1
        return act_to_take, outputs

    def get_str(self):
      addstr = '{}'.format(self.trtype)
      if self.sampling == 'max': addstr += '_max'
      if self.use_aff == 'yes' and not self.cav:
        file_name = '{:010d}_{:03d}_{:03d}{}'.format(FLAGS.snapshot, self.reset_freq, self.latent_reset_freq, addstr)
      elif self.use_aff == 'no':
        file_name = '{:010d}_{:03d}_{:03d}{}_random'.format(FLAGS.snapshot, self.reset_freq, self.latent_reset_freq, addstr)
      elif self.use_aff[0:2] == 'th':
        file_name = '{:010d}_{:03d}_{:03d}{}_aff{}'.format(FLAGS.snapshot, self.reset_freq, self.latent_reset_freq, \
                addstr, self.use_aff)
      elif self.use_aff == 'afmax':
        file_name = '{:010d}_{:03d}_{:03d}{}_afmax'.format(FLAGS.snapshot, self.reset_freq, self.latent_reset_freq, addstr)
      elif self.cav:
        file_name = '{:010d}_{:03d}_{:03d}{}_cav'.format(FLAGS.snapshot, self.reset_freq, self.latent_reset_freq, addstr)
      return file_name

def extract_list_int(l):
    return [int(i) for i in l.split(',')]

def extract_list_float(l):
    return [float(i) for i in l.split(',')]

def get_env(test_env, num_inits):
  cm = ConfigManager()
  cfg_str = 'bs{:d}_N0en1_30_8_16_18_1____mp3d_vp0______TN0_forward_demonstartion_____dilate1_multi1'.format(num_inits) + \
    '.v0_ns40_sn5_frz0_bn1_dr64_one_fsynth_dnc2_gru_demon' + \
    '.dlw1e1_rlw1en1_ent0e0_lr1en4_adam2' + \
    '+bench_'+test_env
  args = cm.process_string(cfg_str)
  args.task.env_task_params.batch_size = num_inits
  args.task.env_task_params_2.batch_size = num_inits
  env = args.env_multiplexer(args.task, 0, 1)
  return env
  
def get_model(model_type, model_dir, snapshot, num_operators, device):
  if model_type == 'our':
    latent_reset_freq = 9 
    reset_freq = 9
    save_path = model_dir
    policy = OurModel(snapshot, save_path, device=device, num_ops=num_operators,
      reset_f=reset_freq, latent_rf=latent_reset_freq, sampling = FLAGS.sampling,
      trtype = FLAGS.trtype)
  
  elif model_type == 'diayn':
    latent_reset_freq = 9
    reset_freq = 9
    save_path = model_dir
    policy = DIAYN(snapshot, save_path, device=device, num_ops=num_operators,
      reset_f=reset_freq, latent_rf=latent_reset_freq, sampling = FLAGS.sampling)
  
  elif model_type == 'our_cav':
    latent_reset_freq = 9 
    reset_freq = 9
    save_path = model_dir
    policy = OurModel(snapshot, save_path, device=device, num_ops=num_operators,
      reset_f=reset_freq, latent_rf=latent_reset_freq, cav = True, sampling = FLAGS.sampling,
      trtype = FLAGS.trtype)

  elif model_type[0:6] == 'our_th':
    latent_reset_freq = 9 
    reset_freq = 9
    save_path = model_dir
    policy = OurModel(snapshot, save_path, device=device, num_ops=num_operators,
            reset_f=reset_freq, latent_rf=latent_reset_freq, use_aff = model_type[4:], sampling = FLAGS.sampling,
      trtype = FLAGS.trtype)


  elif model_type == 'our_noaf':
    latent_reset_freq = 9 
    reset_freq = 9
    save_path = model_dir
    policy = OurModel(snapshot, save_path, device=device, num_ops=num_operators,
      reset_f=reset_freq, latent_rf=latent_reset_freq, use_aff = 'no', sampling = FLAGS.sampling,
      trtype = FLAGS.trtype)
      

  elif model_type == 'our_afmax':
    latent_reset_freq = 9 
    reset_freq = 9
    save_path = model_dir
    policy = OurModel(snapshot, save_path, device=device, num_ops=num_operators,
      reset_f=reset_freq, latent_rf=latent_reset_freq, use_aff = 'afmax', sampling = FLAGS.sampling,
      trtype = FLAGS.trtype)

  elif model_type == 'curiosity':
    reset_freq = 0
    save_path = model_dir
    policy = Curiosity(snapshot, save_path, device=device,
      reset_f=reset_freq, sampling = FLAGS.sampling)

  elif model_type == 'stat':
    rng_stat = np.random.RandomState(0)
    policy = StatModel(rng_stat, p_forward=FLAGS.p_forward)

  elif model_type == 'random':
    rng_rand = np.random.RandomState(0)
    policy = RandomModel(rng_rand)
  
  elif model_type == 'collav':
    rng_rand = np.random.RandomState(0)
    policy = CollavModel(rng_rand)
  
  return policy


def run_model(env, policy, device, num_inits=8, num_orients=5, num_runs=100, unroll_length=80, render_flag = True):
  rng_env = np.random.RandomState(0)
  rng_init = np.random.RandomState(0)
  rng_orient = np.random.RandomState(9)
  
  twopi = 2.0*3.14
  random_orientations = np.arange(num_orients) * twopi
  random_orientations = random_orientations / num_orients
  randor = FLAGS.randor
  
  save_dir = os.path.join(FLAGS.logdir_prefix, FLAGS.expt_name, FLAGS.test_env)
  utils.mkdir_if_missing(save_dir)
  method_str = policy.get_str()
  standard_str = 'n{:04d}_inits{:02d}_or{:02d}_unroll{:03d}_rinit{:01d}'.format(num_runs, num_inits, num_orients, unroll_length,randor)
  file_name = standard_str + '.' + method_str + '.json'
  save_path_str = os.path.join(save_dir, file_name)
  
  if os.path.exists(save_path_str) and not FLAGS.rerun:
    logging.error('Reading benchmarks from %s.', save_path_str)
    with open(save_path_str, 'r') as f:
      dd = json.load(f)
    print(dd)
    return None
  
  file_name = standard_str + '.' + method_str + '.pkl'
  save_path_str = os.path.join(save_dir, file_name)
  logging.error('Writing benchmarks to %s.', save_path_str)

  all_states = []
  all_action_prob = []
  all_gt_action = []
  teacher_action = []
  init_states = [] 
  for i in tqdm(range(num_runs), desc=method_str):
      e, eid = env.sample_env(rng_env)
      init_env_state = e.reset(rng_env)

      if randor:
        random_orientations = rng_orient.rand(num_orients)*2.0*np.pi
      
      #sampling point in room
      init_env_state = [
        e.task.env._sample_point_on_map(np.random.RandomState(rng_init.randint(1e6)), 
          in_room=True)]
      init_states.append(list(init_env_state))
      states_i = []
      action_prob_i = []
      gt_action_i = []
      for orient_n in random_orientations:
          state = list(init_env_state[0])
          state[2] = orient_n
          states = [[list(state) for _ in range(num_inits)]]
          action_prob = []
          gt_action = []

          #### Resetting Policy for the next episode
          policy.reset()
          collision_f = [False for cli in range(num_inits)]
          for j in range(unroll_length):
              if render_flag:
                  img = e.task.env.render_views(states[j])
                  img_device = []
                  for _ in range(num_inits):
                    img_device.append(data_transforms['val'](img[0][_]).reshape([1,1,3,224,224]).to(device))
                  img_device = torch.cat(img_device, 0)
              else:
                  img_device = None

              ### Take Action Policy
              with torch.no_grad():
                  act_to_take, outputs = policy.act(img_device, collision_f, j==0)
              
              new_state, reward = e.take_action(states[j], act_to_take, j)
              collision_f = [False for cli in range(num_inits)]
              for itr_c in range(num_inits):
                  if (np.array(states[-1]) == np.array(new_state)).all() and act_to_take[itr_c] > 0:
                      collision_f[itr_c] = True
              states.append(new_state)
              action_prob.append(outputs)
              gt_action.append(act_to_take)
          states_i.append(states)
          action_prob_i.append(action_prob)
          gt_action_i.append(gt_action)
      all_states.append(states_i)
      all_action_prob.append(action_prob_i)
      all_gt_action.append(gt_action_i)
  save_things(save_path_str, all_states, all_action_prob, all_gt_action, init_states)
  tts_base = load_pkls([save_path_str])
  tts_base = process_tts(tts_base)
  metrics = compute_metrics_i(tts_base[0], e, n=num_runs)
  file_name = standard_str + '.' + method_str + '.json'
  keys = ['expt_name', 'logdir_prefix', 'num_inits', 'num_runs', 'num_orients',
    'unroll_length', 'model_type', 'test_env']
  for k in keys:
    metrics[k] = getattr(FLAGS, k)
  
  out_file_name = os.path.join(save_dir, file_name)
  logging.error('Writing metrics to %s.', out_file_name)
  with open(out_file_name, 'w') as f:
    json.dump(metrics, f, sort_keys=True, separators=(',', ': '), indent=4)
    print(json.dumps(metrics, sort_keys=True, separators=(',', ': '), indent=4))

  return None

def save_things(save_path_str, all_states, all_action_prob, all_gt_action, init_states):
  all_states = np.array(all_states)
  all_action_prob = np.array(all_action_prob)
  all_gt_action = np.array(all_gt_action)
  init_states = np.array(init_states)
  
  all_states = np.transpose(all_states, [0,1,3,2,4])
  sh = all_states.shape
  all_states = np.reshape(all_states, [sh[0], sh[1]*sh[2], sh[3], 1, sh[4]])
  
  all_action_prob = np.transpose(all_action_prob, [0,1,3,2,4])
  sh = all_action_prob.shape
  all_action_prob = np.reshape(all_action_prob, [sh[0], sh[1]*sh[2], sh[3], 1, sh[4]])
  
  all_gt_action = np.transpose(all_gt_action, [0,1,3,2])
  sh = all_gt_action.shape
  all_gt_action = np.reshape(all_gt_action, [sh[0], sh[1]*sh[2], sh[3], 1])

  utils.save_variables(save_path_str, 
    [all_states, all_action_prob, all_gt_action, init_states],
    ['states','act_prob', 'gt_action', 'init'], overwrite=True)
  # tt = utils.load_variables(save_path_str)
  # # (100, 40, 81, 1, 3)
  # # (100, 40, 80, 1, 4)
  # # (100, 40, 80, 1)
  # # (100, 1, 3)
  # for v in tt.values():
  #   print(v.shape)
 
def set_flags_and_run(**kwargs):
  for k in kwargs.keys():
    assert(hasattr(FLAGS, k))
    setattr(FLAGS, k, kwargs[k])
  worker()

def worker():
  test_env = FLAGS.test_env 
  env = get_env(test_env, FLAGS.num_inits)
  FLAGS.randor = FLAGS.randor > 0
  assert(len(env.envs) == 1)
  if FLAGS.stable_mdt:
      trav = env.envs[0].task.env.task.road
      trav_area = np.nonzero(trav)[0].shape[0]
      FLAGS.unroll_length = int(np.sqrt(trav_area))
  device = torch.device("cuda:{:d}".format(FLAGS.gpu_id) if torch.cuda.is_available() else "cpu")
  model_dir = os.path.join(FLAGS.logdir_prefix, FLAGS.expt_name)
  assert(os.path.exists(model_dir)), "model_dir does not exist: {:s}".format(model_dir)
  policy = get_model(FLAGS.model_type, model_dir, FLAGS.snapshot, FLAGS.num_operators, device)
  render_flag = FLAGS.model_type not in ['stat', 'random', 'collav']
  run_model(env, policy, device, FLAGS.num_inits, FLAGS.num_orients,
    FLAGS.num_runs, FLAGS.unroll_length, render_flag)

def main(_):
  worker()

if __name__ == '__main__':
  app.run(main)
