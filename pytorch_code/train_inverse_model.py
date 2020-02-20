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
from pytorch_code.model_utils import InverseNet_RN, Conditional_Net, Resnet18_c, ActionConditionalLSTM, InverseNet, InverseNetEF_RN
from pytorch_code.train_utils import Data_Bank, SS_Explore, FPVImageDataset, GenSSDataset_MT, GenPairedDataset, overlay_text, parse_args 
from torchvision import datasets, models, transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import visdom
import cv2
import matplotlib.pyplot as plt
import argparse
from pytorch_code.train_utils import Logger

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
    return [int(i) for i in l.split(',')]

def extract_list_float(l):
    return [float(i) for i in l.split(',')]

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

cmd_parser = argparse.ArgumentParser(description='Process some integers.')
cmd_parser.add_argument('--expt_args', type=str, help='args for experiment', default = '_')
cmd_parser.add_argument('--num_workers', type=int, help='num_workers')
cmd_parser.add_argument('--init_run_no', type=int, default = 0, help='initialize from')
cmd_args = cmd_parser.parse_args()
num_workers = cmd_args.num_workers
expt_parser = parse_args([('step_size','8'), ('nori','12'), ('ss_steps',30), \
        ('batch_size',32), ('dTheta','0,0'), ('dH','0,0'), ('LF', 'T'),('nstarts',-1)],\
        [extract_list_int, extract_list_int, int, int, extract_list_float, extract_list_float, str, int])
args_dict = expt_parser(cmd_args.expt_args)
save_path = os.path.join('output/mp3d/invmodels',cmd_args.expt_args,'')

EPOCH = 100
step_size = args_dict['step_size']
batch_size=args_dict['batch_size']
ss_steps = args_dict['ss_steps']
vis = visdom.Visdom(env = cmd_args.expt_args)

st1 = 'bs1_N2en1_4_'
st2 = '_16_18_1____mp3d_vp0______TN0_forward_demonstartion_____dilate1_multi1__'
st3 = '.v0_ns40_sn5_frz0_bn1_dr64_one_fsynth_dnc2_gru_demon.dlw1e1_rlw1en1_ent0e0_lr1en4_adam2+train_train1'

args = generate_args([st1, st2, st3], [args_dict['step_size'], args_dict['nori']], [1,3], np.random.RandomState(0), max(1,num_workers))
#env = args.env_multiplexer(args.task, 0, 1)
rng = np.random.RandomState(0)

## NN initialization
if args_dict['LF'] == 'T':
    inv_model = InverseNet_RN()
elif args_dict['LF'] == 'F':
    inv_model = InverseNetEF_RN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inv_model.to(device)
init_run_no = cmd_args.init_run_no

if init_run_no>0:
    inv_model.load_state_dict(torch.load(save_path+'{0:03d}'.format(init_run_no)+'_inv_model.pth'))

gQ = Queue()
for i in range(max(1,num_workers)):
    gQ.put(i)

dataset = GenSSDataset_MT(env_args = args, transform = data_transforms['train'], \
        ss_steps = ss_steps, mpqueue = gQ, num_workers = max(1, num_workers),\
        dH = args_dict['dH'], dTheta = args_dict['dTheta'], nstarts = args_dict['nstarts'])


#save_path = os.path.join('output/systest/'+'ss_inverse_models/', '{:04d}'.format(ss_steps) , '')
if not os.path.exists(save_path):
    os.makedirs(save_path)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cmd_args.num_workers)

init_lr = 1e-3
optimizer = optim.Adam(list(inv_model.parameters()), lr=init_lr)#, momentum=0.9)
if args_dict['LF'] ==  'F':
    for param in inv_model.resnet_l5.parameters():
        param.requires_grad = False
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000,20000,40000,80000,150000], gamma=0.1)
criterion = nn.CrossEntropyLoss()
total=0
correct=0

for iter in range(init_run_no):
    scheduler.step()

'''
win_name_acc = vis.scatter([[1,1]])
win_name_dist = vis.scatter([[1,1]])
win_name_sample = vis.images([np.random.random([3, 10, 10])])
plt.plot([1, 23, 2, 4])
win_name_error = vis.matplot(plt)
win_name_total_acc = vis.matplot(plt)
'''
log_freq = 20
expt_logger = Logger(['inv_loss', 'inv_acc', 'acc_hist', 'dist'], save_path, init_run_no, log_freq = log_freq)
for epoch in range(EPOCH):
    running_loss = []
    running_acc = []
    for i, sampled_batch in enumerate(dataloader):

        if (i + init_run_no) == 40000 and args_dict['LF'] == 'F':
            for param in inv_model.resnet_l5.parameters():
                param.requires_grad = True

        optimizer.zero_grad()
        img0_device = sampled_batch['img0'].to(device).reshape([-1,3,224,224])
        img1_device = sampled_batch['img1'].to(device).reshape([-1,3,224,224])
        act_device = sampled_batch['act'].to(device)
        outputs = inv_model(img0_device, img1_device)
        loss = criterion(outputs, act_device.reshape([-1]))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i%log_freq == 0:
            pred_act = torch.max(outputs, 1)[1]
            correct_pred = (pred_act == act_device.reshape([-1])).cpu().detach().numpy()
            act_cpu = sampled_batch['act'].reshape([-1])
            hist_i = []
            dist_i = []
            for hi in range(4):
                idx_i = np.where(act_cpu == hi)
                hist_i.append([hi,np.sum(correct_pred[idx_i])*1.0/len(idx_i[0])])
                dist_i.append([hi, len(idx_i[0])])
            vis.scatter(hist_i, win = 'win_name_acc')
            vis.scatter(dist_i, win = 'win_name_dist')

            vis.line(X = [init_run_no + i], Y = [loss.item()], \
                    win = 'Loss',update = 'append', opts = dict(title='Loss'))
            vis.line(X = [init_run_no + i], Y = [100.0*np.mean(correct_pred)], \
                    win = 'Acc',update = 'append', opts = dict(title='Acc'))
            # 'inv_loss, inv_acc, acc_hist, dist'
            expt_logger.update([(init_run_no+i, loss.item()), \
                    (init_run_no + i, 100.0*np.mean(correct_pred)),\
                    hist_i, dist_i])
            pred_act_reshaped = pred_act.cpu().detach().reshape(act_device.shape).numpy()
            title_list = []
            viz_samples = 4
            viz_len = 10
            cnt_i = 0
            for tmp_i, tmp_j in zip(\
                    sampled_batch['act'][0:viz_samples, 0:viz_len].reshape([-1]), \
                    pred_act_reshaped[0:viz_samples, 0:viz_len].reshape([-1])):
                title_list.append('gt:'+str(int(tmp_i)) + '_p:' + str(int(tmp_j)))

            disp_imgs = overlay_text(sampled_batch['oimg'][0:viz_samples,0:viz_len].reshape([-1,3, 224, 224]).numpy()\
                    , title_list)
            vis.images(disp_imgs, win = 'win_name_sample', nrow = viz_len)


        if i%1000 == 0:
            torch.save(inv_model.state_dict(), save_path+'{0:03d}'.format(init_run_no + i)+'_inv_model.pth')

torch.save(inv_model.state_dict(), save_path+'final_inv_model.pth')
import pdb; pdb.set_trace()
