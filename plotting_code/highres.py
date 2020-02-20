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
run_no = cmd_args.run_no
save_path = os.path.join(cmd_args.expt_args, cmd_args.test_type, 'jct_viz', cmd_args.plot_type, str(run_no), '')
def save_jpeg_i(data, pth):
    cv2.imwrite(pth[:-3]+'jpg',data[:,:,::-1])

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

def save_images(im, save_id):
    img_save_path = os.path.join(save_path, 'images_hd',cmd_args.save_img, save_id)
    #save_img_i(im, img_save_path)
    save_jpeg_i(im, img_save_path)

if cmd_args.save_img is not None:
    utils.mkdir_if_missing(os.path.join(save_path,'images_hd',cmd_args.save_img,''))

cmd_args = cmd_parser.parse_args()
cm = ConfigManager()
test_type = cmd_args.test_type#'area3'#sys.argv[1]#'val'
file_name = cmd_args.expt_args#sys.argv[3]#'fwd_002_pTrue.pth'
args = cm.process_string('bs1_N0en1_'+cmd_args.path_length+'_20_16_18_1____mp3dHD_vp0______TN0_forward_demonstartion_____dilate1_multi1.v0_ns40_sn5_frz0_bn1_dr64_one_fsynth_dnc2_gru_demon.dlw1e1_rlw1en1_ent0e0_lr1en4_adam2+bench_'+test_type)
env = args.env_multiplexer(args.task, 0, 1)
rng_env = np.random.RandomState(0)
text_file = os.path.join(save_path, 'images',cmd_args.save_img, 'coords.txt')
txtfile = open(text_file)
e, eid = env.sample_env(rng_env)
for l in txtfile:
    fname = l.split(':')[0]
    pos = l.split(':')[1].split(',')
    pos = [float(itr) for itr in pos]
    inp_images_o = e.task.env.render_views([pos])[0]
    save_images(inp_images_o[0], fname)
