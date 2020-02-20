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
from pytorch_code.model_utils import Conditional_Net, Resnet18_c, Conditional_Net_RN, ActionConditionalLSTM, Latent_Dist_RN, \
         Conditional_Net_RN, Latent_Dist_RN, Latent_NetV, Conditional_Net_RN5, Latent_Dist_RN5, InverseNetEF_RN
from pytorch_code.train_utils import Data_Bank, SS_Explore, FPVImageDataset, TrajectoryDataset, \
        parse_args
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import scipy

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

cmd_parser = argparse.ArgumentParser(description='Process some integers.')
cmd_parser.add_argument('--expt_args', type=str, help='args for experiment')
cmd_parser.add_argument('--run_no', type=int, default = 0, help='initialize from')
cmd_parser.add_argument('--test_type', type=str, default = '', help='test area')
cmd_parser.add_argument('--num_operators', type=int, default = 4, help='test area')
cmd_parser.add_argument('--path_length', type=str, default = '10', help='test area')
cmd_parser.add_argument('--plot_type', type=str, default = 'entropy', help='test area')
cmd_parser.add_argument('--save_img', type=str, default = None, help='test area')
cmd_parser.add_argument('--traj', type=str, default = 'expert', help='trajectory or random')

cmd_args = cmd_parser.parse_args()
cm = ConfigManager()
test_type = cmd_args.test_type#'area3'#sys.argv[1]#'val'
file_name = cmd_args.expt_args#sys.argv[3]#'fwd_002_pTrue.pth'
args = cm.process_string('bs1_N0en1_'+cmd_args.path_length+'_20_16_18_1____mp3d_vp0______TN0_forward_demonstartion_____dilate1_multi1.v0_ns40_sn5_frz0_bn1_dr64_one_fsynth_dnc2_gru_demon.dlw1e1_rlw1en1_ent0e0_lr1en4_adam2+bench_'+test_type)
env = args.env_multiplexer(args.task, 0, 1)
num_operators = cmd_args.num_operators
latent_dist = Latent_Dist_RN5(num_operators)

run_no = cmd_args.run_no
load_path = cmd_args.expt_args + '/'
latent_dist.load_state_dict(torch.load(load_path + '{0:03d}'.format(run_no)+'_latent_dist.pth'))

rng_env = np.random.RandomState(0)

save_path = os.path.join(cmd_args.expt_args, cmd_args.test_type, 'jct_viz', cmd_args.plot_type, str(run_no), '')
if not os.path.exists(save_path):   os.makedirs(save_path)
if cmd_args.save_img is not None:
    utils.mkdir_if_missing(os.path.join(save_path,'images',cmd_args.save_img,''))

for param in latent_dist.parameters():
    param.require_grad = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
latent_dist.to(device)



def get_ms_size(ent):
    c = None; colormap = None
    if cmd_args.plot_type in 'entropy': 
        s_size = np.exp(2*ent)/100
    elif cmd_args.plot_type == 'entropy_th': 
        s_size = np.exp(ent*2)/10
    elif cmd_args.plot_type[0:2] == 'op':
        s_size = 50*ent
        #c = ent
    elif cmd_args.plot_type[0:4] == 'mxop':
        s_size = 20*ent
    elif cmd_args.plot_type == 'two_way':
        s_size = np.exp(15.0*ent)/20
        #c = ent
        #colormap = 'bwr'
    return s_size, c, colormap

def get_entropy(dist_pred):
    if cmd_args.plot_type == 'entropy': 
        entropy = scipy.stats.entropy(dist_pred.transpose(), base = 2)
    elif cmd_args.plot_type == 'entropy_th' : 
        entropy = np.sum(dist_pred>=0.15,1)
    elif cmd_args.plot_type[0:2] == 'op' : 
        op_id = int(cmd_args.plot_type[2])
        entropy = np.exp(5.0*dist_pred[:,op_id])/100.0#>0.5
    elif cmd_args.plot_type[0:4] == 'mxop':
        op_id = int(cmd_args.plot_type[4])
        entropy = 1 * np.argmax(dist_pred,1) == op_id
    elif cmd_args.plot_type == 'two_way' : 
        entropy = 0.5 - np.abs(0.5-dist_pred[:,1]/(dist_pred[:,1] + dist_pred[:,2]))

    return entropy

def save_img_i(data, pth):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(data)
    plt.savefig(pth, dpi = height) 
    plt.close()

def save_images(imgs, states, flags, run_i):
    text_file = os.path.join(save_path, 'images',cmd_args.save_img, 'coords.txt')
    txtf = open(text_file,'a')
    for n, im, f in zip(range(len(flags)), imgs, flags):
        img_save_path = os.path.join(save_path, 'images',cmd_args.save_img, '{0:02d}_{1:02d}.pdf'.format(run_i, n)) 
        if f: 
            stl = [str(itrval) for itrval in states[n]]
            save_img_i(im, img_save_path) 
            txtf.write('{0:02d}_{1:02d}.pdf:'.format(run_i, n) + ','.join(stl)+'\n')
    txtf.close()


def setup_plots(ax, full_view):
    ax.imshow(1-full_view.astype(np.float32)/255., vmin=-0.5, vmax=1.5, cmap='Greys', origin='lower')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_axis_off()

def gen_save_plot(tv, xyt, pth, run_i, s_size, c, colormap = None):
    ax = plt.gca()
    setup_plots(ax, tv)
    xmax = np.max(xyt[:,1])+30; xmin = np.min(xyt[:,1])-30
    ymax = np.max(xyt[:,0])+30; ymin = np.min(xyt[:,0])-30
    U = 0.01*np.sin(xyt[:,2]); V = 0.01*np.cos(xyt[:,2])
    Q = ax.quiver(xyt[:,1], xyt[:,0], U, V,  units='width', width = 0.002, scale = 0.6)
    ax.plot(xyt[0,1], xyt[0,0], 'r*', markersize = 20)
    ax.plot(xyt[-1,1], xyt[-1,0], 'g*', markersize = 20)
    ax.scatter(xyt[:,1], xyt[:,0], s = s_size, alpha = 0.7, c = c, cmap = colormap)
    ax.legend(['Start', 'Goal'], loc = (100, -10))
    #plt.xlim(xmin-50, xmax+50)
    #plt.ylim(ymin-50, ymax+50)
    plt.savefig(os.path.join(pth, '{:03d}.pdf'.format(run_i)))
    plt.clf()

def filter_ent(xyt, ent, c):
    xyt_ = [xyt[0]]
    ent_ = [ent[0]]
    for e,s in zip(ent[1:], xyt[1:]):
        if (np.array(s[0:2]) == np.array(xyt_[-1][0:2])).all():
            if ent_[-1] < e:
                xyt_[-1] = s; ent_[-1] = e
        else:
            xyt_.append(s); ent_.append(e)
    if c is not None: c = ent_
    return np.array(xyt_), np.array(ent_), c


            
def save_entropy_plot(tv, xyt, ent, pth, run_i):
    s_size, c, colormap = get_ms_size(ent)
    xyt, ent, c = filter_ent(xyt, ent, c)
    #gen_save_plot(tv, xyt, pth, run_i, 10, c = ent, colormap = 'gnuplot')
    gen_save_plot(tv, xyt, pth, run_i, s_size, c, colormap)

def sample_trajectory(e_i):
    init_env_state = e_i.reset(rng_env)
    input = e_i.get_common_data()
    input = e_i.pre_common_data(input)
    return input['teacher_xyt'][0]

def sample_random_points(e_i):
    import pdb; pdb.set_trace()

top_view_path = 'output/active_eval_fwd_models/full_view_' + cmd_args.test_type + '.png'
top_view = cv2.imread(top_view_path)[:,:,0]
for i in range(100):
    e, eid = env.sample_env(rng_env)
    if cmd_args.traj == 'expert':
        states_xyt = sample_trajectory(e)
    elif cmd_args.traj == 'random':
        states_xyt = sample_random_points(e)
    inp_images_o = e.task.env.render_views(states_xyt)[0]
    inp_images = [data_transforms['val'](img_i) for img_i in inp_images_o]
    inp_images = torch.stack(inp_images).to(device)
    with torch.no_grad():
        dist_pred = torch.nn.Softmax()(latent_dist(inp_images)).cpu().detach().numpy()
    entropy = get_entropy(dist_pred)
    if cmd_args.save_img is not None:
        if cmd_args.save_img[0] == 'h':
            save_flag = entropy > float(cmd_args.save_img[1:])
        elif cmd_args.save_img[0] == 'l':
            save_flag = entropy < float(cmd_args.save_img[1:])
        save_images(inp_images_o, states_xyt, save_flag, i)

    save_entropy_plot(top_view, states_xyt, entropy, save_path, i)
    
    #sampling point in room
