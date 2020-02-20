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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
import cv2
import os
import pandas as pd
import skimage.io as skio
import matplotlib.pyplot as plt
from tqdm import tqdm
import base64
import tempfile
import subprocess
import select

font                   = cv2.FONT_HERSHEY_PLAIN
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2


def freezeRSGrad(inpmodel, flag):
    for param in inpmodel.resnet_l5.parameters():
        param.requires_grad = flag

class Logger():
    def __init__(self, var_list, save_path, init = 0, log_freq = 1):
        self.var_list = var_list
        self.paired_values = [[] for itr in range(len(var_list))]
        self.save_path = save_path
        init = init // log_freq
        if init > 0: 
            loaded_vars = utils.load_variables(os.path.join(self.save_path, 'metrics.pkl'))
            self.paired_values = [loaded_vars[n_i][0:init] for n_i in self.var_list]

    def update(self, vals):
        for n,v in enumerate(vals): self.paired_values[n].append(v)
        self.save()

    def save(self):
        utils.save_variables(os.path.join(self.save_path, 'metrics.pkl'), \
                self.paired_values, self.var_list, overwrite = True)
        


def overlay_text(imgs, text_list):
    return_list = []
    for i in range(len(imgs)):
        img_T = np.transpose(imgs[i],[1,2,0])
        img_i = np.zeros(img_T.shape)
        img_i[:,:] = (img_T[:,:]*255).astype('uint8')
        img_i = cv2.putText(img_i, text_list[i], (20, 20), font, 1.2, (255, 0, 0), 2)
        return_list.append(np.transpose(img_i,[2,0,1]))
    return return_list

class GenSSDataset_xyt(Dataset):
    """First Person View Images dataset."""

    def __init__(self, env, transform, ss_steps, rng = None):
        self.transform = transform
        self.ss_steps = ss_steps
        self.env = env
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(0)

    def __len__(self):
        return 32000000
    
    def shift_right(self, pred):
        prev_act = np.copy(pred)
        prev_act[1:] = prev_act[0:-1]
        prev_act[0] = 0
        return prev_act

    def to_onehot(self, a, depth):
        oneh_a = np.zeros([len(a), depth])
        oneh_a[range(len(a)), a] = 1
        return oneh_a
    
    def random_steps(self,init_state, e):
        state_list = [init_state[0]]
        act_list = []
        for i in range(self.ss_steps):
            act_list.append(1+self.rng.choice(3))
            state_list.append(e.take_action([state_list[-1]], [act_list[-1]])[0][0])
        return np.array(state_list), np.array(act_list)

    def __getitem__(self, idx):
        img_list = []
        oimg_list = []
        e, eid = self.env.sample_env(self.rng)
        init_env_state = e.reset(self.rng)
        states, pred = self.random_steps(init_env_state, e)
        return {'states':torch.tensor(states), 'pred':torch.tensor(pred), 'eid':torch.tensor([eid])}

class GenInvDataset_MT(Dataset):
    """First Person View Images dataset."""

    def __init__(self, env_args, transform, ss_steps, mpqueue, num_workers, dH, dTheta, sampling_freq, rng = None):
        #cm = ConfigManager()
        #args = cm.process_string(env_args)
        #env = args.env_multiplexer(args.task, 0, 1)
        self.dH = dH
        self.dTheta = dTheta
        self.num_workers = num_workers
        self.mpqueue = mpqueue
        self.env_args = env_args
        env = None
        self.transform = transform
        self.ss_steps = ss_steps
        self.env = env
        self.sampling_freq = sampling_freq
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(0)

    def __len__(self):
        return 32000000
    
    def sample(self, states, action, freq):
        fstate = []
        fact = []
        for i in range(1, len(states)):
            for j in range(freq):
                alpha = j*1./freq
                fstate.append((alpha)*states[i] + (1.0-alpha)*states[i-1])
                fact.append(action[i-1])
        fstate.append(states[-1]); fact.append(action[-1])
        return np.array(fstate)[:len(states)], np.array(fact)[:len(action)]

    def shift_right(self, pred):
        prev_act = np.copy(pred)
        prev_act[1:] = prev_act[0:-1]
        prev_act[0] = 0
        return prev_act

    def to_onehot(self, a, depth):
        oneh_a = np.zeros([len(a), depth])
        oneh_a[range(len(a)), a] = 1
        return oneh_a
    
    def random_steps(self,init_state, e):
        state_list = [init_state[0]]
        act_list = []
        for i in range(self.ss_steps):
            act_list.append(1+self.rng.choice(3))
            state_list.append(e.take_action([state_list[-1]], [act_list[-1]])[0][0])
        return np.array(state_list), np.array(act_list)

    def __getitem__(self, idx):
        if self.env is None:
            self.process_rank = self.mpqueue.get()
            cm = ConfigManager()
            args = cm.process_string(self.env_args[self.process_rank])
            self.env = args.env_multiplexer(args.task, self.process_rank, self.num_workers)
            self.rng = np.random.RandomState(self.process_rank)
            self.perturbs = np.zeros([100, 2])
            print ('--------------------------------------------')
            print ('--------------------------------------------')
            print(self.env_args[self.process_rank])
            print(self.process_rank)
            print ('--------------------------------------------')
            print ('--------------------------------------------')
        #print('sending idx',idx,'from',self.process_rank, 'with', self.perturbs[0])
        dH = self.dH[0]*self.rng.rand() + self.dH[1]
        dTheta = self.dTheta[0]*self.rng.rand() + self.dTheta[1]
        freq = self.sampling_freq[self.rng.choice(len(self.sampling_freq))]
        self.perturbs = np.zeros([100, 2])
        self.perturbs[:,0] = dH
        self.perturbs[:,1] = dTheta
        img_list = []
        oimg_list = []
        e, eid = self.env.sample_env(self.rng)
        init_env_state = e.reset(self.rng)
        exp_input = e.get_common_data()
        exp_input = e.pre_common_data(exp_input)
        states = exp_input['teacher_xyt'][0]
        pred = exp_input['teacher_actions'][0][:-1]
        states, pred = self.sample(states, pred, freq)
        #import pdb; pdb.set_trace()
        #states, pred = self.random_steps(init_env_state, e)
        input_views = e.task.env.render_views(states, perturbs = self.perturbs[0:len(states),:])[0]
        transformed_list = []
        for i in input_views:
            transformed_list.append(self.transform(i))
        input_views = [F.to_tensor(F.to_pil_image(i)) for i in input_views] 
        return {'img0':torch.stack(transformed_list[0:-1]), 'act':torch.tensor(pred), \
                'img1':torch.stack(transformed_list[1:]), 'oimg':torch.stack(input_views[0:-1])}

class GenInvPosDataset_MT(Dataset):
    """First Person View Images dataset."""

    def __init__(self, env_args, transform, ss_steps, mpqueue, num_workers, dH, dTheta, angle_r, step_r, rng = None):
        #cm = ConfigManager()
        #args = cm.process_string(env_args)
        #env = args.env_multiplexer(args.task, 0, 1)
        self.angle_r = np.array(angle_r)/360.0*(2.0*np.pi)
        self.step_r = step_r
        self.dH = dH
        self.dTheta = dTheta
        self.num_workers = num_workers
        self.mpqueue = mpqueue
        self.env_args = env_args
        env = None
        self.transform = transform
        self.ss_steps = ss_steps
        self.env = env
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(0)

    def __len__(self):
        return 32000000
    
    def shift_right(self, pred):
        prev_act = np.copy(pred)
        prev_act[1:] = prev_act[0:-1]
        prev_act[0] = 0
        return prev_act

    def to_onehot(self, a, depth):
        oneh_a = np.zeros([len(a), depth])
        oneh_a[range(len(a)), a] = 1
        return oneh_a
    
    def random_steps_pos(self,init_state, e):
        angle_r = self.angle_r
        step_r = self.step_r
        state_list = [init_state[0]]
        act_list = []
        for i in range(self.ss_steps):
            angle = self.rng.uniform(low = angle_r[0], high = angle_r[1])
            step = self.rng.uniform(low = step_r[0], high = step_r[1])
            state_i = e.task.env.take_action_pos(state_list[-1], angle, step)
            act_list.append((angle, step))
            state_list.append(state_i)
        return np.array(state_list), np.array(act_list).astype('float32')

    def __getitem__(self, idx):
        if self.env is None:
            self.process_rank = self.mpqueue.get()
            cm = ConfigManager()
            args = cm.process_string(self.env_args[self.process_rank])
            self.env = args.env_multiplexer(args.task, self.process_rank, self.num_workers)
            self.rng = np.random.RandomState(self.process_rank)
            self.perturbs = np.zeros([100, 2])
            print ('--------------------------------------------')
            print ('--------------------------------------------')
            print(self.env_args[self.process_rank])
            print(self.process_rank)
            print ('--------------------------------------------')
            print ('--------------------------------------------')
        #print('sending idx',idx,'from',self.process_rank, 'with', self.perturbs[0])
        dH = self.dH[0]*self.rng.rand() + self.dH[1]
        dTheta = self.dTheta[0]*self.rng.rand() + self.dTheta[1]
        self.perturbs = np.zeros([100, 2])
        self.perturbs[:,0] = dH
        self.perturbs[:,1] = dTheta
        img_list = []
        oimg_list = []
        e, eid = self.env.sample_env(self.rng)
        init_env_state = e.reset(self.rng)
        states, pred = self.random_steps_pos(init_env_state, e)
        input_views = e.task.env.render_views(states, perturbs = self.perturbs[0:len(states),:])[0]
        transformed_list = []
        for i in input_views:
            transformed_list.append(self.transform(i))
        input_views = [F.to_tensor(F.to_pil_image(i)) for i in input_views] 
        return {'img0':torch.stack(transformed_list[0:-1]), 'act':torch.tensor(pred), \
                'img1':torch.stack(transformed_list[1:]), 'oimg':torch.stack(input_views[0:-1])}

class GenSSDataset_MT(Dataset):
    """First Person View Images dataset."""

    def __init__(self, env_args, transform, ss_steps, mpqueue, num_workers, dH, dTheta, rng = None, nstarts = -1):
        #cm = ConfigManager()
        #args = cm.process_string(env_args)
        #env = args.env_multiplexer(args.task, 0, 1)
        self.dH = dH
        self.dTheta = dTheta
        self.num_workers = num_workers
        self.mpqueue = mpqueue
        self.env_args = env_args
        env = None
        self.transform = transform
        self.ss_steps = ss_steps
        self.env = env
        self.nstarts = nstarts // num_workers
        self.perturbs_n = []; self.states_n = []; self.pred_n = []; self.eid_n = []
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(0)

    def __len__(self):
        return 32000000
    
    def shift_right(self, pred):
        prev_act = np.copy(pred)
        prev_act[1:] = prev_act[0:-1]
        prev_act[0] = 0
        return prev_act

    def to_onehot(self, a, depth):
        oneh_a = np.zeros([len(a), depth])
        oneh_a[range(len(a)), a] = 1
        return oneh_a
    
    def random_steps(self,init_state, e):
        state_list = [init_state[0]]
        act_list = []
        for i in range(self.ss_steps):
            act_list.append(1+self.rng.choice(3))
            state_list.append(e.take_action([state_list[-1]], [act_list[-1]])[0][0])
        return np.array(state_list), np.array(act_list)
    
    def update_cache(self, states, perturbs, pred, eid):
        self.states_n.append(states)
        self.perturbs_n.append(perturbs)
        self.pred_n.append(pred)
        self.eid_n.append(eid)

    def sample_episode(self):
        if self.nstarts > 0 and len(self.states_n) < self.nstarts:
            states, perturbs, pred, eid = self.sample_new_episode()
            self.update_cache(states, perturbs, pred, eid)
        else:
            sidx = self.rng.choice(len(self.states_n))
            states, perturbs, pred, eid = self.states_n[sidx], self.perturbs_n[sidx], self.pred_n[sidx], self.eid_n[sidx]
        return states, perturbs, pred, eid

    def sample_new_episode(self):
        dH = self.dH[0]*self.rng.rand() + self.dH[1]
        dTheta = self.dTheta[0]*self.rng.rand() + self.dTheta[1]
        perturbs = np.zeros([5000, 2])
        perturbs[:,0] = dH
        perturbs[:,1] = dTheta
        img_list = []
        oimg_list = []
        e, eid = self.env.sample_env(self.rng)
        init_env_state = e.reset(self.rng)
        states, pred = self.random_steps(init_env_state, e)
        return states, perturbs, pred, eid

    def __getitem__(self, idx):
        if self.env is None:
            self.process_rank = self.mpqueue.get()
            cm = ConfigManager()
            args = cm.process_string(self.env_args[self.process_rank])
            self.env = args.env_multiplexer(args.task, self.process_rank, self.num_workers)
            self.rng = np.random.RandomState(self.process_rank)
            print ('--------------------------------------------')
            print ('--------------------------------------------')
            print(self.env_args[self.process_rank])
            print(self.process_rank)
            print ('--------------------------------------------')
            print ('--------------------------------------------')
        #print('sending idx',idx,'from',self.process_rank, 'with', self.perturbs[0])
        states, perturbs, pred, eid= self.sample_episode()
        e = self.env.get_env(eid)
        input_views = e.task.env.render_views(states, perturbs = perturbs[0:len(states),:])[0]
        transformed_list = []
        for i in input_views:
            transformed_list.append(self.transform(i))
        input_views = [F.to_tensor(F.to_pil_image(i)) for i in input_views] 
        return {'img0':torch.stack(transformed_list[0:-1]), 'act':torch.tensor(pred), \
                'img1':torch.stack(transformed_list[1:]), 'oimg':torch.stack(input_views[0:-1])}


class GenSSDataset(Dataset):
    """First Person View Images dataset."""

    def __init__(self, env, transform, ss_steps, rng = None):
        self.transform = transform
        self.ss_steps = ss_steps
        self.env = env
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(0)

    def __len__(self):
        return 32000000
    
    def shift_right(self, pred):
        prev_act = np.copy(pred)
        prev_act[1:] = prev_act[0:-1]
        prev_act[0] = 0
        return prev_act

    def to_onehot(self, a, depth):
        oneh_a = np.zeros([len(a), depth])
        oneh_a[range(len(a)), a] = 1
        return oneh_a
    
    def random_steps(self,init_state, e):
        state_list = [init_state[0]]
        act_list = []
        for i in range(self.ss_steps):
            act_list.append(1+self.rng.choice(3))
            state_list.append(e.take_action([state_list[-1]], [act_list[-1]])[0][0])
        return np.array(state_list), np.array(act_list)

    def __getitem__(self, idx):
        img_list = []
        oimg_list = []
        e, eid = self.env.sample_env(self.rng)
        init_env_state = e.reset(self.rng)
        states, pred = self.random_steps(init_env_state, e)
        input_views = e.task.env.render_views(states)[0]
        transformed_list = []
        for i in input_views:
            transformed_list.append(self.transform(i))
        input_views = [F.to_tensor(F.to_pil_image(i)) for i in input_views] 
        return {'img0':torch.stack(transformed_list[0:-1]), 'act':torch.tensor(pred), \
                'img1':torch.stack(transformed_list[1:]), 'oimg':torch.stack(input_views[0:-1])}

class GenPairedDataset(Dataset):
    """First Person View Images dataset."""

    def __init__(self, env, transform, rng = None):
        self.transform = transform
        self.env = env
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(0)

    def __len__(self):
        return 32000000
    
    def shift_right(self, pred):
        prev_act = np.copy(pred)
        prev_act[1:] = prev_act[0:-1]
        prev_act[0] = 0
        return prev_act
    def to_onehot(self, a, depth):
        oneh_a = np.zeros([len(a), depth])
        oneh_a[range(len(a)), a] = 1
        return oneh_a
        
    def __getitem__(self, idx):
        img_list = []
        oimg_list = []
        e, eid = self.env.sample_env(self.rng)
        init_env_state = e.reset(self.rng)
        input = e.get_common_data()
        input = e.pre_common_data(input)
        states=input['teacher_xyt'][0][0:-1]
        pred = input['teacher_actions'][0,0:-1]
        input_views = e.task.env.render_views(states)[0]
        transformed_list = []
        for i in input_views:
            transformed_list.append(self.transform(i))
        input_views = [F.to_tensor(F.to_pil_image(i)) for i in input_views] 
        return {'img0':torch.stack(transformed_list[0:-1]), 'act':torch.tensor(pred[0:-1]), \
                'img1':torch.stack(transformed_list[1:]), 'oimg':torch.stack(input_views[0:-1])}


class GenOnlineTrajectoryDataset(Dataset):
    """First Person View Images dataset."""
    def __init__(self, transform, device, inv_model, path_length, rng = None):
        self.folder = 'ytdl/snipped/'
        self.path_length = path_length
        self.dataframes = []
        self.dataacts = []
        self.dataoimg = []
        self.transform = transform
        self.device = device
        self.inv_model = inv_model
        self.cutframes = 70
        import os
        for fname in os.listdir(self.folder):
            if fname.endswith('.mp4'):
                of, fl = self.extract_frames(fname) 
                frs, oimg, acts = self.tune_fr(of, fl)
                if frs is not None:
                    self.dataframes += frs
                    self.dataacts += acts
                    self.dataoimg += oimg
        print('Loaded',len(self.dataframes),'instances')
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(0)

    def extract_frames(self, fname):
        cap = cv2.VideoCapture(self.folder + fname)
        original_frames=[]
        frame_list = []
        while(cap.isOpened()):
            ret, frame_in = cap.read()
            if not ret: break
            frame_in = cv2.resize(frame_in,(224, 224))
            frame = self.flip_axes(frame_in)
            im = (self.transform(frame)).type('torch.FloatTensor')
            frame_list.append(im)
            original_frames.append((frame))
        return original_frames, frame_list

    def flip_axes(self, frame_in):
        frame = np.zeros(frame_in.shape, dtype = frame_in.dtype)
        frame[:,:,0] = frame_in[:,:,2]
        frame[:,:,1] = frame_in[:,:,1]
        frame[:,:,2] = frame_in[:,:,0]
        return frame

    def split_vec(self, v):
        split_list = []
        idl = []
        cnt = 0
        while(cnt + self.path_length <= len(v)):
            split_list.append(v[cnt:cnt+self.path_length])
            idl.append(range(cnt, cnt+self.path_length-1))
            cnt += self.path_length-1
        return split_list, idl

    def tune_fr(self, orig_l, frm_l):

        if len(frm_l) < self.cutframes + 12*self.path_length:
            return None, None, None

        opt_jmp = -1
        opt_h = 1
        for jmp_i in [12, 15, 20]:
            pred_l, _idx = self.run_inv_model(jmp_i, frm_l, orig_l)
            h, _ = np.histogram(pred_l, bins = [0,1,2,3,4], normed = True)
            if h[-1] <= opt_h and len(_idx) >= self.path_length:
                opt_h = h[-1]
                opt_jmp = jmp_i
        pred_l, idxlist = self.run_inv_model(opt_jmp, frm_l, orig_l, False)

        idxlist_s, idl = self.split_vec(idxlist)

        return [[frm_l[itr2] for itr2 in itr1] for itr1 in idxlist_s], \
                [[orig_l[itr2] for itr2 in itr1] for itr1 in idxlist_s], \
                [[pred_l[itr2] for itr2 in itr1] for itr1 in idl]


    def run_inv_model(self, jmp_len, frame_list, original_frames, saveim = False):
        cnt = 0
        plot_counter =0
        pred_list = []
        frlist = []
        acts = []
        idxlist = []
        efflag = True
        for itr in range(len(frame_list)):
            if cnt >= self.cutframes and cnt % jmp_len ==0:
                if efflag: idxlist.append(cnt - jmp_len); efflag = False
                im1 = frame_list[cnt - jmp_len]
                idxlist.append(cnt)
                pred = self.inv_model(frame_list[cnt - jmp_len].unsqueeze(0).to(self.device), frame_list[cnt].unsqueeze(0).to(self.device))
                pred = torch.nn.Softmax(1)(pred)
                pred_list.append(np.argmax(pred.cpu().detach().numpy()))
                #pred_list.append([np.round(round_i,2) for round_i in pred.cpu().detach().numpy()])
                if saveim:
                    invplot(original_frames[cnt-jmp_len], original_frames[cnt], pred.cpu().detach().numpy(), plot_counter)
                    plot_counter+=1
            cnt+=1
        return pred_list, idxlist

    def __len__(self):
        return len(self.dataframes)
    
    def shift_right(self, pred):
        prev_act = np.copy(pred)
        prev_act[1:] = prev_act[0:-1]
        prev_act[0] = 0
        return prev_act

    def to_onehot(self, a, depth):
        oneh_a = np.zeros([len(a), depth])
        oneh_a[range(len(a)), a] = 1
        return oneh_a
        
    def __getitem__(self, idx):
        oimg_list = self.dataoimg[idx][0:self.path_length]
        transformed_list = self.dataframes[idx][0:self.path_length]
        pred = self.dataacts[idx][0:self.path_length-1]
        pred = [int(itr) for itr in pred]
        prev_act = self.to_onehot(self.shift_right(pred), 4)
        input_views = [F.to_tensor(F.to_pil_image(i)) for i in oimg_list] 

        return {'img':torch.stack(transformed_list), 'act':torch.tensor(pred), \
                'prev_act':torch.tensor(prev_act).float(), 'oimg':torch.stack(input_views)}

class GenTrajectoryGestures(Dataset):
    """First Person View Images dataset."""

    def __init__(self, transform, path_len, datapath, speaker, stride = None, dsrate = 1, thresh = 0, datatype = 'arms'):
        if stride is None: stride = path_len//2
        self.thresh = thresh
        self.transform = transform
        self.path_len = path_len
        self.datapath = datapath
        self.speaker = speaker
        self.stride = stride
        self.dsrate = dsrate
        self.mean = np.array([[   0.0000, -163.4743, -211.8222, -134.2548,  164.7726,  208.7016, 139.6836],
                [   0.0000,    1.7809,  225.3509,  225.7139,   -4.8451,  213.1106, 241.9903]])
        self.stddev = np.array([[ 1.0000, 10.9743, 30.2957, 54.9824, 11.1841, 35.4767, 97.5212],
                    [ 1.0000,  9.4506, 29.2456, 69.2882,  9.3464, 32.7053, 60.5775]])
        self.pose_path = os.path.join(datapath, speaker, 'keypoints_simple/')
        self.frames_path = os.path.join(datapath, speaker, 'frames/')
        self.df_speaker = pd.read_csv(os.path.join(datapath, speaker + '.csv'))
        print('Loaded data from : {}'.format(os.path.join(datapath, speaker + '.csv')))
        self.data = self.slice_data(self.df_speaker)
        print('Sliced data with plen {0}, stride {1}, dsrate {2} into : {3}'\
                .format(path_len, stride, dsrate, len(self.data)))
        if datatype == 'arms':
            self.idxlist = [itr for itr in range(7)]
        elif datatype == 'wrists':
            self.idxlist = [3, 6]

        #self.get_info(100)

    def slice_data(self, df):
        data = []
        interval_list = df.interval_id.unique()
        for interval in interval_list:
            df_i = df[df['interval_id'] == interval]
            data_i = self.slice_data_i(df_i, interval)
            data += data_i
        return data   

    def slice_data_i(self, df_i, interval):
        data = []
        df_sorted = df_i.sort_values(by = ['frame_id'])
        for itr in range(0, len(df_sorted) - self.path_len*self.dsrate, self.dsrate*self.stride):
            data.append((interval, itr))
        return data

    def __len__(self):
        return len(self.data)

    def unpad_images(self, img, rx, ry):
        nx, ny = img.shape[0:2]
        dnx = int(rx*nx)
        dny = int(ry*ny)
        return img[dnx:-dnx, dny:-dny]

    def plot_keypoints(self, img, keypoints, img_width=1280, img_height=720, alpha_img=0.8, cm=plt.cm.rainbow):
        if img is None:
            img = np.ones([img_width, img_height], dtype=uint8) * 255
        fig = plt.figure(figsize=(6, 4))
        plt.imshow(img, alpha=alpha_img)
        colors = cm(np.linspace(0, 1, keypoints.shape[1]))
        plt.scatter(keypoints[0], keypoints[1], s=30, color=colors)
        ax = fig.get_axes()[0]
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        plt.close()
        return self.unpad_images(np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)[:,:,::-1],0.15, 0.08)

    def process_pose(self, pose):
        for i in range(pose.shape[0]):
            pose[i] = pose[i] - pose[i,0]
        pose = (pose - self.mean)/self.stddev
        return pose

    def get_frame(self, idx):
        frame = []
        df = self.df_speaker[self.df_speaker['interval_id'] == self.data[idx][0]].sort_values(by = ['frame_id'])
        for itr in range(self.data[idx][1], self.data[idx][1] + self.path_len*self.dsrate, self.dsrate):
            dffri = df.iloc[itr]
            frame_i = skio.imread(self.frames_path + dffri['frame_fn'])
            pose_i = np.loadtxt(self.pose_path + dffri['pose_fn'])[:,0:7].astype(np.float32)
            pose_i = pose_i[:,self.idxlist]
            frame.append(self.plot_keypoints(frame_i, pose_i))
        return frame

    def get_pose(self, idx):
        pose = []
        df = self.df_speaker[self.df_speaker['interval_id'] == self.data[idx][0]].sort_values(by = ['frame_id'])
        for itr in range(self.data[idx][1], self.data[idx][1] + self.path_len*self.dsrate, self.dsrate):
            dffri = df.iloc[itr]
            pose_i = np.loadtxt(self.pose_path + dffri['pose_fn'])[:,0:7].astype(np.float32)
            pose.append(self.process_pose(pose_i))
        pose = np.array(pose)
        act = pose[1:] - pose[:-1]
        return pose, act

    def getaction(self, act):
        import pdb; pdb.set_trace()

    def __getitem__(self, idx):
        poses, act = self.get_pose(idx)
        poses = poses[:,:,self.idxlist]
        act = act[:,:,self.idxlist]
        act = act.reshape([act.shape[0],-1])
        poses = poses.reshape([poses.shape[0],-1])
        data_dim = poses.shape[1]
        act_shifted = torch.cat([torch.zeros([1,data_dim]), torch.tensor(act, dtype = torch.float32)])[:-1]
        act_bin = np.copy(act)
        act_bin[np.where(np.abs(act_bin) < self.thresh)] = 0
        act_bin = np.sign(act_bin)
        act_shifted_bin = torch.cat([torch.zeros([1,data_dim]), torch.tensor(act_bin, dtype = torch.float32)])[:-1]
        #return {'frames':torch.tensor(frames), 'poses':torch.tensor(poses)}
        return {'poses':torch.tensor(poses, dtype = torch.float32), \
                'act': torch.tensor(act, dtype = torch.float32), \
                'act_shifted': act_shifted, \
                'act_bin': torch.tensor(act_bin, dtype = torch.float32), \
                'act_shifted_bin': torch.tensor(act_shifted_bin, dtype = torch.float32), \
                'batch_idx': idx}

class GenTrajectoryDataset_MT(Dataset):
    """First Person View Images dataset."""

    def __init__(self, env_args, transform, mpqueue, num_workers, dTheta, dH, path_len, rng = None, nstarts = -1):
        self.mpqueue = mpqueue
        self.num_workers = num_workers
        self.env_args = env_args
        env = None
        self.transform = transform
        self.env = env
        self.dTheta = dTheta
        self.path_len = path_len - 1
        self.dH = dH
        self.action_list = []
        self.state_list = []
        self.eid_list = []
        self.perturb_list = []
        self.nstarts = nstarts // num_workers
        self.perturbs_n = []; self.states_n = []; self.pred_n = []; self.eid_n = []
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(0)

    def __len__(self):
        return 32000000
    
    def shift_right(self, pred):
        prev_act = np.copy(pred)
        prev_act[1:] = prev_act[0:-1]
        prev_act[0] = 0
        return prev_act

    def to_onehot(self, a, depth):
        oneh_a = np.zeros([len(a), depth])
        oneh_a[range(len(a)), a] = 1
        return oneh_a
    
    def split_data(self, states, actions, eid, perturbs):
        stride = int(self.path_len/2)-1
        state_list = []
        action_list = []
        eid_list = []
        perturb_list = []
        for n in range(0, len(states) - self.path_len, stride):
            state_list.append(states[n:n+self.path_len])
            action_list.append(actions[n:n+self.path_len])
            eid_list.append(eid)
            perturb_list.append(perturbs[n:n+self.path_len])
        
        return action_list, state_list, eid_list, perturb_list

    def pop_data(self):
        st = self.state_list[-1]
        act = self.action_list[-1]
        eid = self.eid_list[-1]
        perturbs = self.perturb_list[-1]
        self.state_list = self.state_list[:-1]
        self.action_list = self.action_list[:-1]
        self.eid_list = self.eid_list[:-1]
        self.perturb_list = self.perturb_list[:-1]
        return st, act, eid, perturbs


    def update_cache(self, states, perturbs, pred, eid):
        self.states_n.append(states)
        self.perturbs_n.append(perturbs)
        self.pred_n.append(pred)
        self.eid_n.append(eid)

    def sample_episode(self):
        if self.nstarts == -1:
            states, perturbs, pred, eid = self.sample_new_episode()
        elif self.nstarts > 0 and len(self.states_n) < self.nstarts:
            states, perturbs, pred, eid = self.sample_new_episode()
            self.update_cache(states, perturbs, pred, eid)
        else:
            sidx = self.rng.choice(len(self.states_n))
            states, perturbs, pred, eid = self.states_n[sidx], self.perturbs_n[sidx], self.pred_n[sidx], self.eid_n[sidx]
        return states, perturbs, pred, eid

    def sample_new_episode(self):
        dH = self.dH[0]*self.rng.rand() + self.dH[1]
        dTheta = self.dTheta[0]*self.rng.rand() + self.dTheta[1]
        perturbs = np.zeros([100, 2])
        perturbs[:,0] = dH
        perturbs[:,1] = dTheta
        img_list = []
        oimg_list = []
        e, eid = self.env.sample_env(self.rng)
        init_env_state = e.reset(self.rng)
        input = e.get_common_data()
        input = e.pre_common_data(input)
        states = input['teacher_xyt'][0][0:-1]
        pred = input['teacher_actions'][0,0:-1]
        return states, perturbs, pred, eid


    def __getitem__(self, idx):
        if self.env is None:
            self.process_rank = self.mpqueue.get()
            cm = ConfigManager()
            args = cm.process_string(self.env_args[self.process_rank])
            self.env = args.env_multiplexer(args.task, self.process_rank, self.num_workers)
            print ('--------------------------------------------')
            print ('--------------------------------------------')
            print(self.process_rank)
            print ('--------------------------------------------')
            print ('--------------------------------------------')

        ### Take care of when to generate a new episode
        if len(self.action_list) == 0:
            states_f, perturbs_f, pred_f, eid_f = self.sample_episode()
            action_list, state_list, eid_list, perturb_list = self.split_data(states_f, pred_f, eid_f, perturbs_f)
            self.action_list = action_list
            self.state_list = state_list
            self.eid_list = eid_list
            self.perturb_list = perturb_list
        states, pred, eid, perturbs = self.pop_data()    
        e = self.env.get_env(eid)
        prev_act = self.to_onehot(self.shift_right(pred), 4)
        input_views = e.task.env.render_views(states, perturbs = perturbs[0:len(states),:])[0]
        transformed_list = []
        plot_img = e.task.env.exp_nav.plot(states, states[-1], save_img = False, only_top = True)
        for i in input_views:
            transformed_list.append(self.transform(i))
        input_views = [F.to_tensor(F.to_pil_image(i)) for i in input_views] 
        return {'img':torch.stack(transformed_list), 'act':torch.tensor(pred), \
                'prev_act':torch.tensor(prev_act).float(), 'oimg':torch.stack(input_views), 'top_view':torch.tensor(plot_img)}

class GenTrajectoryDataset(Dataset):
    """First Person View Images dataset."""

    def __init__(self, env, transform, rng = None):
        self.transform = transform
        self.env = env
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(0)

    def __len__(self):
        return 32000000
    
    def shift_right(self, pred):
        prev_act = np.copy(pred)
        prev_act[1:] = prev_act[0:-1]
        prev_act[0] = 0
        return prev_act
    def to_onehot(self, a, depth):
        oneh_a = np.zeros([len(a), depth])
        oneh_a[range(len(a)), a] = 1
        return oneh_a
        
    def __getitem__(self, idx):
        img_list = []
        oimg_list = []
        e, eid = self.env.sample_env(self.rng)
        init_env_state = e.reset(self.rng)
        input = e.get_common_data()
        input = e.pre_common_data(input)
        states=input['teacher_xyt'][0][0:-1]
        pred = input['teacher_actions'][0,0:-1]
        prev_act = self.to_onehot(self.shift_right(pred), 4)

        input_views = e.task.env.render_views(states)[0]
        transformed_list = []
        for i in input_views:
            transformed_list.append(self.transform(i))
        input_views = [F.to_tensor(F.to_pil_image(i)) for i in input_views] 
        return {'img':torch.stack(transformed_list), 'act':torch.tensor(pred), \
                'prev_act':torch.tensor(prev_act).float(), 'oimg':torch.stack(input_views)}

class TrajectoryDataset(Dataset):
    """First Person View Images dataset."""

    def __init__(self, data_bank, env, transform=None, stacking = 1):
        self.data_bank = data_bank 
        self.transform = transform
        self.env = env
        self.stacking = stacking
        self.rendered_images = np.zeros([self.data_bank.counter,224,224,3],dtype = np.uint8)
        for n in range(data_bank.counter):
            st = self.data_bank[n]
            e = self.env.get_env(st[2])
            self.rendered_images[n] = e.task.env.render_views([st[0]])[0][0]

    def __len__(self):
        return self.data_bank.counter/39

    def __getitem__(self, idx):
        low_limit = idx*39
        img_list = []
        oimg_list = []
        for i in range(low_limit, low_limit + 39):
            idx_i = i
            img_xyt = self.rendered_images[idx_i]
            if self.transform:
                img_xyt_t = self.transform(img_xyt[:,:,::-1])
            img_list.append(img_xyt_t)
            oimg_list.append(img_xyt.transpose().swapaxes(1,2))

        act = self.data_bank[low_limit:low_limit+39][1]
        prev_act = np.append([0],act[0:-1])
        one_hot_prev_act = np.zeros([len(prev_act),4])
        one_hot_prev_act[np.arange(len(prev_act)),prev_act] = 1.0 


        return {'img':torch.stack(img_list), 'act':act, 'prev_act':torch.tensor(one_hot_prev_act).float(), 'oimg':oimg_list}

class FPVImageDataset(Dataset):
    """First Person View Images dataset."""

    def __init__(self, data_bank, env, transform=None, stacking = 1):
        self.data_bank = data_bank 
        self.transform = transform
        self.env = env
        self.stacking = stacking
        self.rendered_images = np.zeros([self.data_bank.counter,224,224,3],dtype = np.uint8)
        for n in range(data_bank.counter):
            st = self.data_bank[n]
            e = self.env.get_env(0)#st[2])
            self.rendered_images[n] = e.task.env.render_views([st[0]])[0][0][::-1]

    def __len__(self):
        return self.data_bank.counter

    def __getitem__(self, idx):
        low_limit = int(idx/39)*39
        img_list = []
        for i in range(self.stacking):
            idx_i = max(low_limit,idx - i)
            img_xyt = self.rendered_images[idx_i]
            if self.transform:
                img_xyt = self.transform(img_xyt)
            img_list.append(img_xyt)

        act = self.data_bank[idx][1]

        return {'img':torch.cat(img_list), 'act':act}

class OptionsDataset(Dataset):
    """Options dataset."""

    def __init__(self, data, env, transform=None):
        self.actions = data['actions']
        self.states = data['states']
        self.eid = data['eid']
        self.n_clusters = len(data['actions'])
        self.slice_len = len(self.actions[0][0])
        self.transform = transform
        self.env = env
        self.paired_data = []
        for i in range(len(self.actions)):
            for j in range(len(self.actions[i])):
                self.paired_data.append((i,self.actions[i][j],self.states[i][j],self.eid[i][j]))

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        data_tuple = self.paired_data[idx]
        eid = data_tuple[3]
        e = self.env.get_env(eid)
        img_list = []
        for i in range(len(data_tuple[2])):
            img_i = e.task.env.render_views([data_tuple[2][i]])[0][0][::-1]
            if self.transform:
                img_i = self.transform(img_i)
            img_list.append(img_i)
        c_id = np.zeros(self.n_clusters)
        c_id[data_tuple[0]] = 1.0
        c_id = torch.tensor(c_id).float()
        c_id = c_id.expand([self.slice_len, self.n_clusters]) 
        c_id_numbered = torch.tensor(data_tuple[0]).long()
         
        return {'img':torch.stack(img_list), 'act':torch.tensor(data_tuple[1]).long(), 'cluster_id':c_id, 'cid_numbered': c_id_numbered}

class Data_Bank:
    def __init__(self, data_dims, label_dims, rng):
        self.train = np.zeros([1000000]+data_dims)
        self.label = np.zeros([1000000]+label_dims).astype(np.int64)
        self.eid = np.zeros(1000000).astype(np.int32)
        self.rng = np.random.RandomState(rng.randint(1e6))
        self.counter = 0

    def insert(self,data,label,eid):
        counter=self.counter
        num_points = len(data)
        self.train[counter:(counter+num_points)] = data
        self.label[counter:(counter+num_points)] = label
        self.eid[counter:(counter+num_points)] = eid
        self.counter += num_points
    
    def __getitem__(self, key):
        return [self.train[key], self.label[key], self.eid[key]]

    def get(self, num_points):
        counter = self.counter
        rand_sample = self.rng.randint(0,counter,num_points)
        return [self.train[rand_sample], self.label[rand_sample], self.eid[rand_sample]]

    def dump_data(self,dump_dir):
        print('Saving ',self.counter, 'instances')
        utils.save_variables(dump_dir, \
                [self.train[0:self.counter], self.label[0:self.counter], self.eid[0:self.counter]], \
                ['train','label', 'eid'], overwrite=True)

    def load_data(self,dump_dir):
        load_var = utils.load_variables(dump_dir)
        train = load_var['train']
        label = load_var['label']
        eid = load_var['eid']
        assert (len(train) == len(label))
        self.counter = len(train)
        self.train[0:self.counter] = train
        self.label[0:self.counter] = label
        self.eid[0:self.counter] = eid
        print('loaded ',self.counter, 'instances')


class SS_Explore:
    def __init__(self, env,init_state):
        self.env = env
        self.last_state = init_state

    def gen_ss_data(self, steps):
        current_state=self.last_state
        state_pairs=[]
        labels=[]
        for i in range(steps):
            act_to_take = np.random.randint(4)
            new_state,reward = e.take_action(current_state,[act_to_take])
            state_pairs.append((current_state[0],new_state[0]))
            labels.append(act_to_take)
            current_state = np.copy(new_state)
        self.last_state = current_state
        return [np.array(state_pairs),np.array(labels)]
    
    def set_state(self,init_state):
        self.last_state = init_state

class parse_args:
    def __init__(self,args_tpl, ops):
        self.arg_dict = {}
        self.args_name = []
        for n,i in enumerate(args_tpl):
            self.args_name.append(i[0])
            if i[1] is not None:
                self.arg_dict[i[0]] = ops[n](i[1])
            else:
                self.arg_dict[i[0]] = None
        self.ops = ops

    def __call__(self,args):
        args = args.split('_')
        for i,j,k in zip(args, self.args_name, self.ops):
            if not(i == ''):
                self.arg_dict[j] = k(i)
        return self.arg_dict
        

def generate_args(st, var, pos, rng, cnt):
    rand_init = []
    for i in range(cnt):
        rinit_i = []
        for j in range(len(var)):
            ridx = rng.randint(len(var[j]))
            rinit_i.append(var[j][ridx])
        rand_init.append(rinit_i)
    ret_strs = []
    for i in range(cnt):
        ret_strs.append(concat_str(st, rand_init[i], pos))
    return ret_strs



def encode(tensor, framerate):
    L = tensor.size(0)
    H = tensor.size(1)
    W = tensor.size(2)

    t = tempfile.NamedTemporaryFile(suffix='.mp4')

    command = [ 'ffmpeg',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '{}x{}'.format(W, H), # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', str(framerate), # frames per second
        '-i', '-', # The imput comes from a pipe
        '-pix_fmt', 'yuv420p',
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', 'h264',
        '-f', 'mp4',
        '-y', # overwrite
        t.name
        ]

    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    output = bytes()

    frame = 0

    print("Encoding...")

    with tqdm(total=L) as bar:
        while frame < L:
            state = proc.poll()
            if state is not None:
                print('Could not call ffmpeg (see above)')
                raise IOError

            read_ready, write_ready, _ = select.select([proc.stdout], [proc.stdin], [])

            if proc.stdout in read_ready:
                buf = proc.stdout.read1(1024 * 1024)
                output += buf

            if proc.stdin in write_ready:
                proc.stdin.write(tensor[frame].numpy().tobytes())
                frame += 1
                bar.update()

        remaining_output, _ = proc.communicate()
        output += remaining_output

    data = open(t.name, 'rb').read()
    t.close()

    videodata = """
        <video controls>
            <source type="video/mp4" src="data:video/mp4;base64,{}">
            Your browser does not support the video tag.
        </video>
    """.format(base64.b64encode(output).decode('utf-8'))

    return videodata



#output = encode(video)
#print('Got output: {} bytes'.format(len(output)))
#vis.text(text=videodata, opts=dict(title='Sequence {}'.format(i)))

