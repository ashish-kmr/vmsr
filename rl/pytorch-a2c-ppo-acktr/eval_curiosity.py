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
from pytorch_code.model_utils import Conditional_Net, Resnet18_c, ActionConditionalLSTM, ActionEmbedLSTM
from pytorch_code.train_utils import Data_Bank, SS_Explore, FPVImageDataset, TrajectoryDataset
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import gym
from model import Policy

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

class stat_model:
    def __init__(self, rng, e):
        self.rng = np.random.RandomState(rng.randint(1e6))
        act_list=[]
        for i in range(10000):
            e,eid = env.sample_env(self.rng)
            init_env_state = e.reset(self.rng)
            input = e.get_common_data()
            input = e.pre_common_data(input)
            act_list.append(input['teacher_actions'])
        act_list = np.array(act_list).reshape([1,-1])
        self.aff = np.histogram(act_list,bins=[0,1,2,3,4])[0].astype(np.float)
        self.aff = self.aff/np.sum(self.aff)
        

    def eval(self, renderer, states, all_states):
        aff = self.aff
        act_to_take = [self.rng.choice(a=4,p=aff)]
        return act_to_take, aff

class random_model:
    def __init__(self, rng):
        self.rng = np.random.RandomState(rng.randint(1e6))
    
    def eval(self, renderer, states, all_states):
        aff = 0.25*np.array([1.,1.,1.,1.])
        act_to_take = [self.rng.choice(a=4,p=aff)]
        return act_to_take, aff

class trained_model:
    def __init__(self, rng, model, stack, device):
        self.rng = np.random.RandomState(rng.randint(1e6))
        self.device = device
        self.model = model
        self.stack = stack
    
    def eval(self, renderer, states, all_states):
        img_list = []
        idx = len(all_states)-1
        low_limit = 0
        for i in range(self.stack):
            idx_i = max(low_limit,idx - i)
            img_xyt = renderer(all_states[idx_i])[0][0]
            img_xyt = data_transforms['val'](img_xyt)
            img_list.append(img_xyt)
        inputA = torch.unsqueeze(torch.cat(img_list),0)
        inputA = inputA.to(device)
        outputs = torch.nn.functional.softmax(self.model(inputA))
        aff = outputs[0].cpu().detach().numpy()
        act_to_take = [self.rng.choice(a=4,p=aff)]
        return act_to_take, aff

## ENV initialization
cm = ConfigManager()
import sys
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_type = 'area4'#sys.argv[1]#'val'
model_type = 'trained'#sys.argv[2]#'trained'
file_name = ''#sys.argv[3]#'fwd_002_pTrue.pth'
args = cm.process_string('bs1_N0en1_30_8_16_18_1____mp3d_vp0______TN0_forward_demonstartion_____dilate1_multi1.v0_ns40_sn5_frz0_bn1_dr64_one_fsynth_dnc2_gru_demon.dlw1e1_rlw1en1_ent0e0_lr1en4_adam2+bench_'+test_type)
env = args.env_multiplexer(args.task, 0, 1)
rng = np.random.RandomState(0)
act_rng = np.random.RandomState(0)
## NN initialization
pretrained=True
stacking=1
feature_dim = 512
hidden_lstm_size = 512
env_obs = gym.spaces.dict_space.Dict({"view":gym.spaces.Box(0.,255.,shape=[3,224,224], dtype = np.float32)})
env_act = gym.spaces.Discrete(4)
curiosity = Policy(env_obs, env_act,
       base_kwargs={
          'recurrent': True,
          'hidden_size': 512,
          'policy_type': 'Nav',
          'init_policy': False} ) 
#curiosity = torch.load('/home/sgupta/affordances/code/outputs/curiosity/v2/Curiosity-bs8_sz8_o12_0_16_n16_n0x05_10_10_1_200_dense2_trainfwd-v0/a2c_rnn_Nav/models/modelA.pt')[0]
curiosity = torch.load('/home/sgupta/affordances/code/outputs/curiosity/v2/Curiosity-bs8_sz8_o12_0_16_n16_n0x05_10_10_1_200_dense2_trainfwd-v0/a2c_rnn_Nav/models/modelA.pt')[0]
curiosity.to(device)
curiosity.base.to(device)

rng_init = np.random.RandomState(0)
rng_orient = np.random.RandomState(9)
twopi = 2.0*3.14
random_orientations = [0, twopi/5.0, 2.0*twopi/5.0, 3.0*twopi/5.0, 4.0*twopi/5.0]#3.14*2.0*rng_orient.rand(n_or)
n_or = len(random_orientations)
print(random_orientations)
model_list = {}

all_states = []
all_action_prob = []
all_gt_action = []
teacher_action = []
init_states = []
reset_freq = 100
for i in range(100):
    e, eid = env.sample_env(rng)
    init_env_state = e.reset(rng)
    #sampling point in room
    init_env_state = [e.task.env._sample_point_on_map(np.random.RandomState(rng_init.randint(1e6)),in_room=True)]
    init_states.append(init_env_state)
    states_i = []
    action_prob_i = []
    gt_action_i = []
    for orient_n in random_orientations:
        for n_r in range(8):
            input = e.get_common_data()
            input = e.pre_common_data(input)
            states = [init_env_state]
            states[0][0][2] = orient_n
            action_prob = []
            gt_action = []
            #actions=input['teacher_actions'][0][0:-1]
            act_taken = torch.tensor([[[0.0,0.0,0.0,0]]])
            action = [0]
            hidden_state = torch.zeros([1,hidden_lstm_size]).to(device)
            for j in range(80):
                #act_to_take, aff = model_list[model_type].eval(e.task.env.render_views, states[j], states)
                img = e.task.env.render_views(states[j])
                img_device = data_transforms['val'](img[0][0]).reshape([1,1,3,224,224]).to(device)
                sz0, sz1 = img_device.shape[0:2]
                if j%reset_freq==0:
                    hidden_state = hidden_state*0.0
                    act_taken = torch.tensor([[[0.0,0.0,0.0,0]]])
		value, action, action_log_prob, hidden_state = curiosity.act({'view':img_device[0]},hidden_state,torch.ones(hidden_state.shape).to(device))
                act_taken = torch.zeros([1,1,4])
                act_taken[0,0,action[0,0,0]]=1.0
                new_state, reward = e.take_action(states[j], action[0,0], j)
                states.append(new_state)
                gt_action.append(action[0,0,0].cpu().numpy())
            states_i.append(states)
            action_prob_i.append(action_prob)
            gt_action_i.append(gt_action)
    all_states.append(states_i)
    all_action_prob.append(action_prob_i)
    all_gt_action.append(gt_action_i)

    #all_states = np.array(all_states)
    #all_action_prob = np.array(all_action_prob)
    #all_gt_action = np.array(all_gt_action)
    #init_states = np.array(init_states)
    save_path_str = '/home/ashish/landmark/affordances/output/mp3d/curiosity/area4/A.pkl'
    save_dir = '/'.join(save_path_str.split('/')[:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    utils.save_variables(save_path_str, \
            [np.array(all_states), np.array(all_action_prob), np.array(all_gt_action), np.array(init_states) ], \
            ['states','act_prob', 'gt_action', 'init'], overwrite=True)

