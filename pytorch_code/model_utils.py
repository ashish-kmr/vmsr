from _logging import logging
import torch
import numpy as np
from multiprocessing import Process, Queue
import threading
from six.moves.queue import Queue as LocalQueue
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
import torchvision

class Options(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super(Options, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.rnn = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    
    #img_features should be batch x seq_len x feature_dim. img_fetures should have the class id concatenated
    def forward(self, img_features):
        outputs_, hn = self.rnn(img_features.permute([1,0,2]))
        return self.out(outputs_.permute([1,0,2]).reshape([-1,self.hidden_size]))


class Gestures_Latent_Net(nn.Module):
    def __init__(self, inp_ch, output_dim, pathlen, nl):
        super(Gestures_Latent_Net, self).__init__()
        osh = pathlen - 1
        conv_list = []
        kernel_size = []
        init_flag = True
        input_sz=inp_ch
        self.conv1op = nn.Conv1d(input_sz, 32, 5, padding = (5 - 1)//2)
        self.conv2op = nn.Conv1d(32, 32, 5, padding = (5 - 1)//2)
        while(osh > 1):
            conv_list.append(32)
            kernel_size.append(5)
            osh = osh // 2
        self.conv_ops = nn.ModuleList()
    
        input_sz = 32
        for i,j in zip(conv_list, kernel_size):
            self.conv_ops.append(nn.Conv1d(input_sz, i, j, padding = (j - 1)//2))
            input_sz = i

        self.pool = nn.MaxPool1d(2, 2)
        self.last_ch = conv_list[-1]
        self.fc1 = nn.Linear(osh * self.last_ch, output_dim)
        self.osh = osh
        if nl == 'relu':
            self.nl = F.relu
        elif nl == 'tanh':
            self.nl = torch.nn.Tanh()
            
    def forward(self, x):
        x = self.nl(self.conv1op(x))
        x = self.nl(self.conv2op(x))
        for op in self.conv_ops:
            x=self.pool(self.nl(op(x)))
        x = x.view(-1, self.osh * self.last_ch)
        x = self.fc1(x)
        return x


class ActionConditionalLSTM_c(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActionConditionalLSTM_c, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    
    #img_features should be batch x seq_len x feature_dim. img_fetures should have the class id concatenated
    def forward(self, img_features, hidden_state):
        outputs_, hn = self.rnn(img_features, hidden_state)
        return outputs_, hn


class ActionEmbedFCLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embed_list, fc_inp, embed_dim):
        super(ActionEmbedFCLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_list = embed_list
        self.rnn = nn.GRU(input_size +  ((1 + len(embed_list)) * embed_dim), hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.fc_embed = nn.Linear(fc_inp, embed_dim)
        self.embed_layers = nn.ModuleList([nn.Embedding(i, embed_dim) for i in embed_list])
    
    #img_features should be batch x seq_len x feature_dim. img_fetures should have the class id concatenated
    def forward(self, img_feat, embed_inps, fc_inps,  hidden_state):
        embed_features = []
        for i in range(len(embed_inps)):
            embed_features.append(self.embed_layers[i](embed_inps[i]))
        embed_fc_feat = self.fc_embed(fc_inps) 
        img_features = torch.cat([img_feat] + embed_features + [embed_fc_feat], 2) 
        outputs_, hn = self.rnn(img_features.permute([1,0,2]), hidden_state)
        return self.out(outputs_.permute([1,0,2]).reshape([-1,self.hidden_size])), hn


class ActionEmbedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embed_list, embed_dim):
        super(ActionEmbedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_list = embed_list
        self.rnn = nn.GRU(input_size + (len(embed_list) * embed_dim), hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.embed_layers = nn.ModuleList([nn.Embedding(i, embed_dim) for i in embed_list])
    
    #img_features should be batch x seq_len x feature_dim. img_fetures should have the class id concatenated
    def forward(self, img_feat, embed_inps, hidden_state):
        embed_features = []
        for i in range(len(embed_inps)):
            embed_features.append(self.embed_layers[i](embed_inps[i]))
        img_features = torch.cat([img_feat] + embed_features, 2) 
        outputs_, hn = self.rnn(img_features.permute([1,0,2]), hidden_state)
        return self.out(outputs_.permute([1,0,2]).reshape([-1,self.hidden_size])), hn

class ActionConditionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len = None):
        super(ActionConditionalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.rnn = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    
    #img_features should be batch x seq_len x feature_dim. img_fetures should have the class id concatenated
    def forward(self, img_features, hidden_state):
        outputs_, hn = self.rnn(img_features.permute([1,0,2]), hidden_state)
        return self.out(outputs_.permute([1,0,2]).reshape([-1,self.hidden_size])), hn

class MetaController(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MetaController, self).__init__()
        self.inp_l = nn.Linear(input_size, hidden_size)
        self.nl = nn.ReLU()
        self.out_l = nn.Linear(hidden_size, output_size)
        #feedforward/no rnn
            
    def forward(self, x1):
        return (self.out_l(self.nl(self.inp_l(x1))))


class InverseNetEF_RN(nn.Module):
    def __init__(self):
        super(InverseNetEF_RN, self).__init__()
        resnet = models.resnet18(pretrained = True)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])
        self.conv1 = nn.Conv2d(1024, 512, kernel_size = 3, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(512, 128, kernel_size = 3, padding = 1, bias = False)
        self.fc1 = nn.Linear(128, 4)
        self.pool = nn.MaxPool2d(2, 2)
            
    def forward(self, x1,x2):
        x1=self.resnet_l5(x1)
        x2=self.resnet_l5(x2)
        x=torch.cat((x1,x2),1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,128)
        x = self.fc1(x)
        return x

class InverseNet_RN(nn.Module):
    def __init__(self):
        super(InverseNet_RN, self).__init__()
        self.resnet = models.resnet18(pretrained = True)
        self.resnet.fc = nn.Linear(512,256)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)

            
    def forward(self, x1,x2):
        x1=self.resnet(x1)
        x2=self.resnet(x2)
        x1 = x1.view(-1, 256)
        x2 = x2.view(-1, 256)
        x=torch.cat((x1,x2),1)
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class InverseNet(nn.Module):
    def __init__(self):
        super(InverseNet, self).__init__()
        conv_list=[16, 32, 64, 64, 64, 128]
        self.conv_ops = nn.ModuleList()
        input_sz=3
        for i in conv_list:
            self.conv_ops.append(nn.Conv2d(input_sz, i, 3))
            input_sz=i
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2 * 1 * 1 * 128, 4)
        #self.fc2 = nn.Linear(2048, 512)
        #self.fc3 = nn.Linear(512, 128)
        #self.fc4 = nn.Linear(128, 4)

            
    def forward(self, x1,x2):
        for op in self.conv_ops:
            x1=self.pool(F.relu(op(x1)))
        for op in self.conv_ops:
            x2=self.pool(F.relu(op(x2)))
        x1 = x1.view(-1, 1 * 1 * 128)
        x2 = x2.view(-1, 1 * 1 * 128)
        x=torch.cat((x1,x2),1)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc1(x)
        return x

class Resnet18_c(nn.Module):
    def __init__(self, pretrained, stacking, output_size = 4):
        super(Resnet18_c, self).__init__()
        self.fwd_net=models.resnet18(pretrained=pretrained)
        conv_0 = nn.Conv2d(stacking*3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        for i in range(stacking):
            conv_0.weight.data[:,i*3:(i+1)*3] = self.fwd_net.conv1.weight.data/(1.0*stacking)
        self.fwd_net.conv1 = conv_0
        num_ftrs = self.fwd_net.fc.in_features
        self.fwd_net.fc = nn.Linear(num_ftrs, output_size) 
            
    def forward(self, x):
        return self.fwd_net(x)

class Latent_Dist_RN5_c(nn.Module):
    def __init__(self, output_dim, pretrained = True):
        super(Latent_Dist_RN5_c, self).__init__()
        resnet = models.resnet18(pretrained = pretrained)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])
        self.conv1 = nn.Conv2d(512, 256, kernel_size = 3, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size = 3, padding = 1, bias = False)
        self.fc1 = nn.Linear(128, output_dim)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x=self.resnet_l5(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,128)
        return x

class Latent_Dist_RN5(nn.Module):
    def __init__(self, output_dim):
        super(Latent_Dist_RN5, self).__init__()
        resnet = models.resnet18(pretrained = True)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])
        self.conv1 = nn.Conv2d(512, 256, kernel_size = 3, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size = 3, padding = 1, bias = False)
        self.fc1 = nn.Linear(128, output_dim)
        self.pool = nn.MaxPool2d(2, 2)
            
    def forward(self, x):
        x=self.resnet_l5(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,128)
        x = self.fc1(x)
        return x

class Latent_Dist_RN(nn.Module):
    def __init__(self, output_dim):
        super(Latent_Dist_RN, self).__init__()
        self.resnet = models.resnet18(pretrained = True)
        self.resnet.fc = nn.Linear(512,output_dim)
            
    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(x)
        return x

class Conditional_Net_RN5N(nn.Module):
    def __init__(self, feat_dim = 2048, pretrained = True):
        super(Conditional_Net_RN5N, self).__init__()
        resnet = models.resnet18(pretrained = pretrained)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])
        self.conv1 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1, bias = True)
        #self.conv2 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1, bias = True)
        self.fc1 = nn.Linear(512, feat_dim)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
            
    def forward(self, x):
        x=self.pool(self.resnet_l5(x))
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,512)
        x = self.fc1(x)
        return x

class Conditional_Net_RN5(nn.Module):
    def __init__(self, feat_dim = 2048):
        super(Conditional_Net_RN5, self).__init__()
        resnet = models.resnet18(pretrained = True)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])
        self.conv1 = nn.Conv2d(512, 256, kernel_size = 3, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias = False)
        self.fc1 = nn.Linear(256, feat_dim)
        self.pool = nn.MaxPool2d(2, 2)
            
    def forward(self, x):
        x=self.resnet_l5(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,256)
        x = self.fc1(x)
        return x

class Conditional_Net_RN(nn.Module):
    def __init__(self, feat_dim = 2048):
        super(Conditional_Net_RN, self).__init__()
        self.resnet = models.resnet18(pretrained = False)
        self.resnet.fc = nn.Linear(512,feat_dim)
            
    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(x)
        return x

class Gesture_Featurizer(nn.Module):
    def __init__(self, input_sz = 14, feat_dim = 128):
        super(Gesture_Featurizer, self).__init__()
        fc_list = nn.ModuleList()
        fc_sizes = [128, 256]
        inp_ch = input_sz
        for out_sz in fc_sizes:
            fc_list.append(nn.Linear(inp_ch,out_sz))
            inp_ch = out_sz
        self.fc_list = fc_list
        self.fc_final = nn.Linear(fc_sizes[-1],feat_dim)
            
    def forward(self, x):
        for op in self.fc_list:
            x=F.relu(op(x))
        x = self.fc_final(x)
        return x

class Conditional_Net(nn.Module):
    def __init__(self, feat_dim = 2048):
        super(Conditional_Net, self).__init__()
        conv_list=[16,32,64,128]
        self.conv_ops = nn.ModuleList()
        input_sz=3
        for i in conv_list:
            self.conv_ops.append(nn.Conv2d(input_sz, i, 3))
            input_sz=i
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12 * 12 * 128, feat_dim)
        # self.fc2 = nn.Linear(2048, 1024)
        # self.fc3 = nn.Linear(1024, 512)
        # self.fc4 = nn.Linear(512, 4)

            
    def forward(self, x):
        for op in self.conv_ops:
            x=self.pool(F.relu(op(x)))
        x = x.view(-1, 12 * 12 * 128)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
        return x

class Latent_NetV(nn.Module):
    def __init__(self, inp_ch, output_dim, pathlen):
        super(Latent_NetV, self).__init__()
        osh = pathlen
        conv_list = []
        kernel_size = []
        #for i in range(pathlen-6):
        init_flag = True
        while(osh > 4):
            osh = (osh -2)
            if init_flag:
                conv_list.append(32)
                init_flag = False
            else:
                conv_list.append(64)
            kernel_size.append(3)
        osh = osh // 2
        self.conv_ops = nn.ModuleList()
        input_sz=inp_ch

        for i,j in zip(conv_list, kernel_size):
            self.conv_ops.append(nn.Conv1d(input_sz, i, j))
            input_sz = i

        self.pool = nn.MaxPool1d(2, 2)
        self.last_ch = conv_list[-1]
        self.fc1 = nn.Linear(osh * self.last_ch, output_dim)
        self.osh = osh

            
    def forward(self, x):
        for op in self.conv_ops:
            x=F.relu(op(x))
        x = self.pool(x)
        x = x.view(-1, self.osh * self.last_ch)
        x = F.relu(self.fc1(x))
        return x

class Latent_Net(nn.Module):
    def __init__(self, inp_ch, output_dim, conv_list = None, kernel_size = None):
        super(Latent_Net, self).__init__()
        if conv_list is None:
            conv_list=[32,64]
        if kernel_size is None:
            kernel_size = [3,3]
        self.conv_ops = nn.ModuleList()
        input_sz=inp_ch
        for i,j in zip(conv_list, kernel_size):
            self.conv_ops.append(nn.Conv1d(input_sz, i, j))
            input_sz = i
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(2 * 64, output_dim)

            
    def forward(self, x):
        for op in self.conv_ops:
            x=F.relu(op(x))
        x = self.pool(x)
        x = x.view(-1, 2 * 64)
        x = F.relu(self.fc1(x))
        return x
