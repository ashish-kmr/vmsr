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
from pytorch_code.model_utils import InverseNet_RN, Conditional_Net, Resnet18_c, \
        ActionConditionalLSTM, Latent_Net, Conditional_Net_RN, Latent_Dist_RN, \
        Latent_Dist_RN5, Conditional_Net_RN5, InverseNetEF_RN, \
        Conditional_Net_RN5N, Latent_NetV

from pytorch_code.train_utils import Data_Bank, SS_Explore, FPVImageDataset, \
        GenTrajectoryDataset_MT, overlay_text, parse_args, freezeRSGrad, Logger

from torchvision import datasets, models, transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import visdom
import cv2
import matplotlib.pyplot as plt
import sys
from env.mp_env import take_freespace_action
import argparse
import time

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

def concat_str(st, var, pos):
    st_i = 0
    var_i = 0
    concat_str = []
    for i in range(len(var) + len(st)):
        if i in pos:
            concat_str.append(str(var[var_i]))
            var_i+=1
        else:
            concat_str.append(st[st_i])
            st_i+=1
    return ''.join(concat_str)


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



def to_onehot(agpu, depth):
    a = agpu.cpu().numpy()
    oneh_a = np.zeros([len(a), depth])
    oneh_a[range(len(a)), a] = 1
    return oneh_a

def compute_metrics(pred_act, gt_act):
    act_cpu = gt_act.cpu()
    correct_pred = (pred_act == gt_act).cpu().detach().numpy()
    hist_i = []
    dist_i = []
    for hi in range(4):
        idx_i = np.where(act_cpu == hi)
        hist_i.append([hi,np.sum(correct_pred[idx_i])*1.0/len(idx_i[0])])
        dist_i.append([hi, len(idx_i[0])])

    return hist_i, dist_i, 100*np.mean(correct_pred)

def concat_strs(a, b, c, d):
    c_max = np.argmax(c.detach().cpu().numpy(), 1)
    d_max = np.argmax(d.detach().cpu().numpy(), 1)
    title_list = []
    for n, tmp_i, tmp_j in (zip(range(a.reshape([-1]).size()[0]), a.reshape([-1]), b.reshape([-1]))):
        if n%a.size()[1] == 0:
            title_list.append('gt:'+str(int(tmp_i)) + '_p:' + str(int(tmp_j)) \
                    + '_cp:' + str(int(c_max[int(n/a.size()[1])])) \
                    + '_cgt:' + str(int(d_max[int(n/a.size()[1])])) )
        else:
            title_list.append('gt:'+str(int(tmp_i)) + '_p:' + str(int(tmp_j)))
    return title_list

def rollout(act_list):
    init_env_state=[[980,200,0]]
    rollout_states = []
    states = [init_env_state]
    lstep_size = 8
    for j in range(len(act_list)):
        act = act_list[j]
        new_state = take_freespace_action(angle_value, lstep_size, states[j], [act], j)
        states.append(new_state)
    return states


def get_latent_plots(z, acts, op_len, cl, ax):
    cl_idx = np.where(z[:,cl] == 1)
    acts_cl = acts[cl_idx[0]]
    plot_clusters(acts_cl, ax, cl)
        

def get_top_view_plot(output_dir,st_ls, full_view, ax, cl):
  line_width=[1,1,1,1,1]
  color_lst=['g-','b-','r-','k-']
  #import pdb; pdb.set_trace()
  ax.imshow(1-full_view.astype(np.float32)/255., vmin=-0.5, vmax=1.5, cmap='Greys', origin='lower')

  max_ = np.max(np.max(st_ls,axis=0),axis=0)[0:2][::-1]
  min_ = np.min(np.max(st_ls,axis=0),axis=0)[0:2][::-1]

  mid_ = (min_+max_)/2.
  sz = np.maximum(1.1*np.max(max_-min_)/2., 100)
  ax.axis('off')
  ax.set_xlim([mid_[0]-sz, mid_[0]+sz])
  ax.set_ylim([mid_[1]-sz, mid_[1]+sz])

  for n, state_i in enumerate(st_ls):
      ax.plot(st_ls[n,:,1], st_ls[n,:,0], color_lst[0], alpha=0.8, lw=line_width[0], ms=line_width[0])
  ax.set_title('ns: ' + str(len(st_ls)) + 'z' + str(cl))
  

def plot_clusters(cluster, ax, cl):
    local_rollout = []
    if len(cluster) == 0 :
        cluster = torch.tensor(np.array([[0]]))
    for i in cluster:
        states = rollout(i)
        local_rollout.append(states)

    full_view = 255 + np.zeros((1112, 524))
    get_top_view_plot(None, np.array(local_rollout)[:,:,0,:], full_view, ax, cl)

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        #transforms.ColorJitter(1,1,1,0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

def extract_list_int(l):
    if l is not None:
        return [int(i) for i in l.split(',')]
    else:
        return None

def extract_list_float(l):
    if l is not None:
        return [float(i) for i in l.split(',')]
    else:
        return None

EPOCH=100

cmd_parser = argparse.ArgumentParser(description='Process some integers.')
cmd_parser.add_argument('--expt_args', type=str, help='args for experiment')
cmd_parser.add_argument('--num_workers', type=int, help='num_workers')
cmd_parser.add_argument('--init_run_no', type=int, default = 0, help='initialize from')
cmd_parser.add_argument('--inv_base', type=str, default = 'output/mp3d/invmodels/', help='base path for inverse model')
cmd_parser.add_argument('--no_viz', action='store_true', default = False, help=' switch off viz')
cmd_args = cmd_parser.parse_args()
num_workers = cmd_args.num_workers
expt_parser = parse_args([('path_length','10'),('step_size','8'), \
        ('inv_model_id','-1'), ('init_path',0), ('batch_size',32), ('num_operators',4), \
        ('decay',10), ('nori','12'), ('dTheta','0,0'), ('dH','0,0'), ('fmodel','sn5'), \
        ('imset','trainfwd'), ('full_len', None), ('nstarts', -1)],\
        [str, extract_list_int, int, int, int, int, int, extract_list_int, \
        extract_list_float, extract_list_float, str, str, extract_list_int, int])

args_dict = expt_parser(cmd_args.expt_args.split('.')[0])
inv_pth = '4,10,16_10,15,20__8_30,-10_60,-30_F'
if len(cmd_args.expt_args.split('.'))>1: 
    inv_args = cmd_args.expt_args.split('.')[1]
    if inv_args == 'None':
        inv_pth = None
    else:
        inv_pth = inv_args

batch_size=args_dict['batch_size']
save_path = os.path.join('output/mp3d',cmd_args.expt_args,'')
step_size = args_dict['step_size']
num_operators = args_dict['num_operators']
n_ori = 12
angle_value = [0,2.0*np.pi/n_ori,-2.0*np.pi/n_ori,0]
decay = args_dict['decay']
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not cmd_args.no_viz:
    vis = visdom.Visdom(env = 'slicing_' + cmd_args.expt_args)
## ENV initialization
path_length = args_dict['path_length']
full_length = args_dict['full_len']
if full_length is None: 
    full_length = [path_length]
cm = ConfigManager()
#plot_args = cm.process_string('bs1_N2en1_'+path_length+'_8_16_18_1____mp3d_vp0______TN0_forward_demonstartion_____dilate1_multi1.v0_ns40_sn5_frz0_bn1_dr64_one_fsynth_dnc2_gru_demon.dlw1e1_rlw1en1_ent0e0_lr1en4_adam2+train_train1')

st0 = 'bs1_N2en1_'
st1 = '_'
st2 = '_16_18_1____mp3d_vp0______TN0_forward_demonstartion_____dilate1_multi1__'
st3 = '.v0_ns40_sn5_frz0_bn1_dr64_one_fsynth_dnc2_gru_demon.dlw1e1_rlw1en1_ent0e0_lr1en4_adam2+train_'+args_dict['imset']
args = generate_args([st0, st1, st2, st3], \
        [full_length, args_dict['step_size'], args_dict['nori']], \
        [1,3,5], np.random.RandomState(0), max(1,num_workers))
#args = 'bs1_N2en1_'+path_length+'_' + str(step_size) + '_16_18_1____mp3d_vp0______TN0_forward_demonstartion_____dilate1_multi1.v0_ns40_sn5_frz0_bn1_dr64_one_fsynth_dnc2_gru_demon.dlw1e1_rlw1en1_ent0e0_lr1en4_adam2+train_trainfwd'
#env = plot_args.env_multiplexer(plot_args.task, 0, 1)
rng = np.random.RandomState(0)
## NN initialization
pretrained=True
stacking=1
feature_dim = 512
hidden_lstm_size = 256
#featurizer=Resnet18_c(pretrained=pretrained, stacking = stacking, output_size = feature_dim)
init_run_no = cmd_args.init_run_no
#latent = Latent_Net(4, num_operators, conv_list = [32,32,32,32,32,32,64], kernel_size = [7,7,7,7,7,3,3])
#latent = Latent_Net(4, num_operators, conv_list = [32,32,32,64], kernel_size = [5,5,5,3])
latent = Latent_NetV(4, num_operators, int(args_dict['path_length'])-2)
latent_dist = Latent_Dist_RN5(num_operators) 
if args_dict['fmodel'] == 'sn5':
    featurizer=Conditional_Net(feat_dim = feature_dim)
elif args_dict['fmodel'] == 'RN':
    featurizer=Conditional_Net_RN(feat_dim = feature_dim)
elif args_dict['fmodel'] == 'RN5':
    featurizer=Conditional_Net_RN5(feat_dim = feature_dim)
elif args_dict['fmodel'] == 'RN5N':
    featurizer=Conditional_Net_RN5N(feat_dim = feature_dim)

model_freeze_list = ['RN5', 'RN5N']

act_con_lstm = ActionConditionalLSTM(feature_dim+4+num_operators, hidden_lstm_size, 4, 39)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
featurizer.to(device)
act_con_lstm.to(device)
latent.to(device)
latent_dist.to(device)
gQ = Queue()
for i in range(max(1,num_workers)):
    gQ.put(i)
dataset = GenTrajectoryDataset_MT(env_args = args,transform = data_transforms['train'], mpqueue = gQ, \
        rng = np.random.RandomState(0), num_workers = max(1,num_workers), dTheta = args_dict['dTheta'], \
        dH = args_dict['dH'], path_len = int(path_length), nstarts = args_dict['nstarts'])

if inv_pth is not None:
    inv_model = InverseNetEF_RN()

    inv_model.load_state_dict(torch.load(\
            os.path.join(cmd_args.inv_base, inv_pth,\
            '{:03d}_inv_model.pth'.format(args_dict['inv_model_id']))\
            ))

    for param in inv_model.parameters():
            param.requires_grad = False
    inv_model.to(device)


if init_run_no>0:
    featurizer.load_state_dict(torch.load(save_path+'{0:03d}'.format(init_run_no)+'_featurizer.pth'))
    act_con_lstm.load_state_dict(torch.load(save_path+'{0:03d}'.format(init_run_no)+'_lstm.pth'))
    latent.load_state_dict(torch.load(save_path+'{0:03d}'.format(init_run_no)+'_latent.pth'))
    latent_dist.load_state_dict(torch.load(save_path+'{0:03d}'.format(init_run_no)+'_latent_dist.pth'))


dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
init_lr = 1e-4
optimizer = optim.Adam(list(latent.parameters()) + list(featurizer.parameters())+list(act_con_lstm.parameters()), lr=init_lr)
optimizer_dist = optim.Adam(list(latent_dist.parameters()), lr= init_lr)

if args_dict['fmodel'] in model_freeze_list:
    freezeRSGrad(featurizer, False)
freezeRSGrad(latent_dist, False)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000,20000,40000,80000,150000], gamma=0.1)
scheduler_dist = optim.lr_scheduler.MultiStepLR(optimizer_dist, milestones=[10000,20000,40000,80000,150000], gamma=0.1)

criterion = nn.CrossEntropyLoss()
total=0
correct=0
log_freq = 20
expt_logger = Logger(['fwd_loss', 'fwd_acc', 'invmodel_acc', 'dist_loss', 'hist_acc', 'dist'], \
        save_path, init_run_no, log_freq = log_freq)
for epoch in range(EPOCH):
    running_loss = []
    running_acc = []
    running_acc_act = []
    running_dist_loss = []
    for i, sampled_batch in enumerate(dataloader):
        if i==0: initial_time = time.time() 

        if (i + init_run_no) == 5000:
            if args_dict['fmodel'] in model_freeze_list:
                freezeRSGrad(featurizer, True)
            freezeRSGrad(latent_dist, True)
        
        optimizer.zero_grad()
        optimizer_dist.zero_grad()
        
        # Images and ground truth actions
        img_device_gt = sampled_batch['img'].to(device)
        act_device_gt = sampled_batch['act'][:,0:-1].to(device)
        
        # Constructing pseudo gt from inverse model
        if inv_pth is not None:
            reshape_sz = [-1,3,224,224]
            act_pred = inv_model(img_device_gt[:,0:-1].reshape(reshape_sz), img_device_gt[:,1:].reshape(reshape_sz))
            act_device = torch.max(act_pred,1)[1].reshape(act_device_gt.shape)
        else:
            act_device = act_device_gt

        # Dropping the last image 
        img_device = img_device_gt[:,0:-1]
        # Constructing prev act one hot from pseudo gt
        act_device_shifted = nn.ConstantPad1d((1,0), 0)(act_device[:,0:-1])
        prev_act_device = torch.tensor(to_onehot(act_device_shifted.reshape([-1]), depth = 4)).reshape(act_device.shape+(4,))
        prev_act_device = prev_act_device.type(torch.FloatTensor).to(device)
         
        # Constructing onehot of actions for latent model
        cact = torch.tensor(to_onehot(act_device.reshape([-1]), depth = 4)).reshape(act_device.shape+(4,)).permute(0,2,1)
        cact_device = cact.type(torch.FloatTensor).to(device)
        zvec = latent(cact_device) 
        if decay > 0: 
            tau = 1.0 + (decay*1.0)/(i + 1.0)
        else:
            tau = 1.0

        zvec_oh = nn.functional.gumbel_softmax(zvec, hard = True, tau = tau)
        
        # forward + backward + optimize
        sz0, sz1 = img_device.shape[0:2]
        features = featurizer(img_device.reshape([-1,3,224,224])).reshape([sz0, sz1, feature_dim])
        rep_zvec = torch.unsqueeze(zvec_oh,1).repeat(1,int(path_length)-2,1)
        concat_features = torch.cat([features,prev_act_device, rep_zvec],2)
        
        hidden_state = torch.zeros([1, sz0, hidden_lstm_size]).to(device)
        
        outputs, hidden_state = act_con_lstm(concat_features, hidden_state)
        dist_pred = latent_dist(img_device[:,0])
      
        loss = criterion(outputs, act_device.reshape([-1]))
        loss_dist = criterion(dist_pred, zvec_oh.max(dim = 1)[1])
         
        loss.backward()
        loss_dist.backward()

        optimizer.step()
        optimizer_dist.step()

        scheduler.step()
        scheduler_dist.step()

        if i%20 == 0:
            delta_time = time.time() - initial_time
            print('--------------average sample time :: {0:03f} -------'.format(delta_time*1.0/(i+1.0)))

        if i%20 == 0 and not cmd_args.no_viz:
            plt.tight_layout()
            plt.cla()
            fig, ax = plt.subplots(2,max(num_operators//2,2),figsize=(8,8))
            for cl_i in range(num_operators):
                ax_i = cl_i%2
                ax_j = cl_i//2
                get_latent_plots(zvec_oh.cpu().detach().numpy(), sampled_batch['act'], num_operators, cl_i, ax[ax_i, ax_j])
             
            vis.matplot(plt, win = 'options')
            plt.close(fig)
            pred_act = torch.max(outputs, 1)[1]
            hist_i, dist_i, acc_i = compute_metrics(pred_act, act_device.reshape([-1]))
            act_hist_i, act_dist_i, act_acc_i = compute_metrics(act_device.reshape([-1]), act_device_gt.reshape([-1]))
            
            # Plot stats for forward learning
            vis.scatter(hist_i, update = 'replace', win = 'win_name_acc', opts = dict(title='Fwd Model Acc/Class'))
            vis.scatter(dist_i, update = 'replace', win = 'win_name_dist', opts = dict(title='Fwd Model Class Dist'))

            # PLot stats for act prediction
            vis.scatter(act_hist_i, update = 'replace', win = 'win_name_acc_act', opts = dict(title='Inv Model Acc/Class'))
            vis.scatter(act_dist_i, update = 'replace', win = 'win_name_dist_act', opts = dict(title='Inv Model Class Dist'))
             
            vis.line(X = [i], Y = [loss.item()], win = 'fwdloss',update = 'append', opts = dict(title='Fwd Loss'))
            vis.line(X = [i], Y = [acc_i], win = 'fwdacc',update = 'append', opts = dict(title='Fwd Acc'))
            vis.line(X = [i], Y = [act_acc_i], win = 'actacc',update = 'append', opts = dict(title='Act Acc'))
            vis.line(X = [i], Y = [loss_dist.item()], win = 'distacc',update = 'append', opts = dict(title='Dist Acc'))

            viz_samples = 4
            
            pred_act_reshaped = pred_act.cpu().detach().reshape(act_device.shape).numpy()
            title_list = concat_strs(act_device[0:viz_samples], \
                    pred_act_reshaped[0:viz_samples], dist_pred[0:viz_samples], zvec_oh[0:viz_samples])

            oimg_list = sampled_batch['oimg'][0:viz_samples,0:-1].reshape([-1,3, 224, 224]).numpy()
            disp_imgs = overlay_text(oimg_list, title_list)
            win_name_sample = vis.images(disp_imgs, win = 'win_name_sample', \
                    opts = dict(caption = title_list), nrow = act_device_gt.shape[1])
            np_top_view = np.repeat(sampled_batch['top_view'].numpy()[:,np.newaxis,:,:],3, axis = 1)
            win_name_topview = vis.images(np_top_view, win = 'win_name_topview', nrow = int((np_top_view.shape[0])**0.5 ))
            expt_logger.update([\
                    (init_run_no+i, loss.item()), \
                    (init_run_no + i, acc_i),\
                    (init_run_no + i, act_acc_i),\
                    (init_run_no + i, loss_dist.item()),\
                    hist_i, dist_i])
            print("forward Acc: {}, Invmodel Acc: {}".format(acc_i, act_acc_i))
        if i%2000 == 0:
            torch.save(featurizer.state_dict(), save_path+'{0:03d}'.format(init_run_no + i)+'_featurizer.pth')
            torch.save(act_con_lstm.state_dict(), save_path+'{0:03d}'.format(init_run_no + i)+'_lstm.pth')
            torch.save(latent.state_dict(), save_path+'{0:03d}'.format(init_run_no+i)+'_latent.pth')
            torch.save(latent_dist.state_dict(), save_path+'{0:03d}'.format(init_run_no+i)+'_latent_dist.pth')
        # print statistics

    #print('[%d] loss: %.3f' % (epoch + 1,  np.mean(np.array(running_loss))*1.0 / (len(running_loss))))

torch.save(featurizer.state_dict(), save_path+'final_featurizer.pth')
torch.save(act_con_lstm.state_dict(), save_path+'final_lstm.pth')
torch.save(latent.state_dict(), save_path+'final_latent.pth')
torch.save(latent_dist.state_dict(), save_path+'final_latent_dist.pth')
#import pdb; pdb.set_trace()
