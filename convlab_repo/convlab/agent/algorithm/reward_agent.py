# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import os
from collections import defaultdict
import logging
from torch.autograd import Variable
import torch.nn as nn
import json
import torch.nn.functional as F
import copy
import sys
import pickle

from .torch_utils import GumbelConnector, LayerNorm

device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

INT = 0
LONG = 1
FLOAT = 2
def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(torch.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.cuda.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    else:
        if dtype == INT:
            var = var.type(torch.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    return var

def load_reward_model(agent, pre_sess_path, use_gpu):
    print(pre_sess_path)
    # if not os.path.isfile(pre_sess_path) and not os.path.isdir(pre_sess_path):
    # if not os.path.isdir(pre_sess_path):
    #     raise ValueError("No reward model was loaded")
    # else:
    reward_path = os.path.join(pre_sess_path, "model_lirl")
    if use_gpu:
        reward_sess = torch.load(reward_path)
    else:
        reward_sess = torch.load(reward_path, map_location='cpu')
    agent.discriminator.load_state_dict(reward_sess['discriminator'])
    agent.vae.load_state_dict(reward_sess["vae"])
    print("Loading reward model finished!")

def binary2onehot(x):
    batch_size, digit_num = len(x), len(x[0])
    if digit_num != 9:
        raise ValueError("check the binary length and the current one is {}".format(digit_num))
    one_hot_matrix = []
    for line in x:
        one_hot = []
        for v in line:
            if v==0:
                one_hot+=[1,0]
            elif v==1:
                one_hot+=[0,1]
            else:
                raise ValueError("illegal onehot input: {}".format(v))
        one_hot_matrix.append(one_hot)
    return one_hot_matrix


def one_hot_embedding(labels, num_classes):
    # print(labels)
    if type(labels)==list:
        labels = torch.LongTensor(labels)
    y = torch.eye(num_classes) 
    return y[labels]  
    

class RewardAgent(nn.Module):
    def __init__(self, use_gpu):
        super(RewardAgent, self).__init__()
        config = None
        self.use_gpu = use_gpu
        self.discriminator = Discriminator(self.use_gpu)
        self.vae = AutoEncoder(self.use_gpu)

    def cast_gpu(self, var):
        if self.use_gpu:
            return var.cuda()
        else:
            return var
    
    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        if type(inputs)==list:
            return cast_type(Variable(torch.Tensor(inputs)), dtype,
                         self.use_gpu)
        return cast_type(Variable(torch.from_numpy(inputs)), dtype,
                         self.use_gpu)
    
    def _int2binary_9(self,x):
        return list(reversed( [(x >> i) & 1 for i in range(9)]))

    def forward(self,batch_feed):        
        state = batch_feed['states']
        action = batch_feed['actions']
        action_list = action.view(-1).tolist()
        action_binary = []
        for act in action_list:
            action_binary.append(self._int2binary_9(int(act)))
        reward = self.discriminator(state, self.np2var(action_binary,FLOAT))
        return reward

"""
How to add a new model? 
reward_path
D_code
\ -> \\
"""
class RewardAgent_EncoderSide(nn.Module):
    def __init__(self, config, use_gpu=False, model_name = "mine", vae_type='autoencoder', update=False, real_data_feed=None):
        super(RewardAgent_EncoderSide, self).__init__()
        self.use_gpu = use_gpu
        import sys
        # self.discriminator = Discriminator_SA(self.use_gpu)
        # load D
        config.use_gpu = self.use_gpu

        if model_name == "mine":
            self.discriminator = WoZDiscriminator(config)
            # self.vae = AE_3(config)
            # self.vae = AE_3_parallel_VAE(config)
            self.vae = AE_3_parallel_VAE_finish(config)

        elif model_name == "ziming":
            self.discriminator = WoZDiscriminator_ziming(config)
            self.vae = VAE(config)


    def cast_gpu(self, var):
        if self.use_gpu:
            return var.cuda()
        else:
            return var
    
    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        if type(inputs)==list:
            return cast_type(Variable(torch.Tensor(inputs)), dtype, self.use_gpu)
        return cast_type(Variable(torch.from_numpy(inputs)), dtype, self.use_gpu)
        
    def _int2binary_9(self,x):
        return list(reversed( [(x >> i) & 1 for i in range(9)]))

    def get_action_rep(self, action_list):
        return one_hot_embedding(action_list, 300)

    def forward(self, batch_feed, surgery):
        state = batch_feed['states']
        action = batch_feed['actions']
        # print(action)
        action_list = action.view(-1).tolist()
        # action_binary = []
        # for act in action_list:
        #     action_binary.append(self._int2binary_9(int(act)))
        # action_binary_onehot = binary2onehot(action_binary)
        # action_data_feed = self.np2var(action_binary_onehot, FLOAT).view(-1, 18)
        action_data_feed = self.get_action_rep(action_list)
        # state_action = torch.cat([self.cast_gpu(state), self.np2var(action_binary,FLOAT)], -1)
        state_action = self.cast_gpu(state)
        embed_rep = self.vae.get_embed(state_action)
        reward = self.discriminator(embed_rep, action_data_feed, surgery)
        # By using detach to avoid back wards update.
        return reward.detach()
    
    def forward_validate(self, batch_feed):
        state = batch_feed['state_convlab']
        action = batch_feed['action_id']
        action_list = action
        # action_binary = []
        # for act in action_list:
        #     action_binary.append(self._int2binary_9(int(act)))
        # action_binary_onehot = binary2onehot(action_binary)
        # action_data_feed = self.np2var(action_binary_onehot, FLOAT).view(-1, 18)
        action_data_feed = self.get_action_rep(action_list)

        state_action = self.np2var(state,FLOAT)
        embed_rep = self.vae.get_embed(state_action)
        reward = self.discriminator(embed_rep, action_data_feed)
        return reward
    
    def update(self, fake_batch_feed):
        return self.discriminator.disc_train(self.vae, fake_batch_feed)

class RewardAgent_StateVaeActionSeg(RewardAgent_EncoderSide):
    def __init__(self, use_gpu=False, vae_type='autoencoder'):
        super(RewardAgent_StateVaeActionSeg, self).__init__(use_gpu, vae_type)
        self.discriminator = WoZDiscriminator_StateVaeActionSeg(use_gpu)
        action_rep_path = './data/multiwoz/action_rep_seg.json'
        if not os.path.isfile(action_rep_path):
            raise ValueError("No action rep was loaded")
        with open(action_rep_path, 'r') as f:
            action_rep_seg = json.load(f)
            assert len(action_rep_seg)==300 
        self.action_rep_seg = self.np2var(action_rep_seg, FLOAT)
    
    def get_action_rep(self, action_list):
        if type(action_list)==list:
            act_index = torch.LongTensor(action_list)
        else:
            act_index = action_list
        return self.action_rep_seg[act_index]
        

        
#########################################################################
###########   The following parts are for the Neural Networks  ##########
#########################################################################
class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.use_gpu = config.use_gpu
        self.flush_valid = False
        self.config = config
        self.kl_w = 0.0

        with open(os.path.join(root_dir, "convlab_repo/convlab/agent/algorithm/mask_python2_onehot.pkl"), 'rb') as f:
            self.mask = pickle.load(f).float()
        self.mask = self.cast_gpu(self.mask)

    def cast_gpu(self, var):
        if self.use_gpu:
            return var.cuda().float()
        else:
            return var.cpu().float()

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        if type(inputs) == list:
            return cast_type(Variable(torch.Tensor(inputs)), dtype,
                             self.use_gpu)
        return cast_type(Variable(torch.from_numpy(inputs)), dtype,
                         self.use_gpu)

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, batch_cnt, loss, retain_graph = False):
        total_loss = self.valid_loss(loss, batch_cnt)
        # total_loss += self.l2_norm()
        total_loss.backward(retain_graph = retain_graph)
        # self.clip_gradient()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = 0.0
        for key, l in loss.items():
            if l is not None:
                total_loss += l
        return total_loss

    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)

    def get_optimizer(self, config):
        if config.op == 'adam':
            print("Use adam")
            return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=config.init_lr, betas=(0.5, 0.999))
        elif config.op == 'sgd':
            print("Use SGD")
            return torch.optim.SGD(self.parameters(), lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            print("RMSProp")
            return torch.optim.RMSprop(self.parameters(), lr=config.init_lr,
                                       momentum=config.momentum)

    def clip_gradient(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.clip)

    def l2_norm(self):
        l2_reg = None
        for W in self.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)
        return l2_reg * self.config.l2_lambda


class Discriminator(BaseModel):
    def __init__(self, use_gpu):
        super(Discriminator, self).__init__(use_gpu)
        dropout = 0.3
        self.state_in_size = 392
        self.action_in_size = 9
        self.state_rep = nn.Linear(self.state_in_size, int(self.state_in_size/2))
        self.action_rep = nn.Linear(self.action_in_size, int(self.action_in_size/2))
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(int(self.state_in_size/2 + self.action_in_size/2), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        state_1 = self.state_rep(self.cast_gpu(state))
        action_1 = self.action_rep(self.cast_gpu(action))
        # print(state.shape, action_1.shape)
        state_action = torch.cat([state_1, action_1], 1)
        validity = torch.sigmoid(self.model(state_action))
        validity = torch.clamp(validity, 1e-6, 1-1e-6)
        return validity

class AutoEncoder(BaseModel):
    def __init__(self, use_gpu):
        super(AutoEncoder, self).__init__(use_gpu)
        self.use_gpu = use_gpu
        # self.vae_in_size = 392 + 300
        self.vae_in_size = 392
        self.vae_embed_size =64
        dropout = 0.3
        
        self.encode_model = nn.Sequential(
            nn.Dropout(dropout),          
            nn.Linear(self.vae_in_size, int(self.vae_in_size/2)),
            nn.Tanh(),
            nn.Dropout(dropout),                      
            nn.Linear(int(self.vae_in_size/2), self.vae_embed_size),
            nn.Tanh(), 
        )
        self.decode_model = nn.Sequential(
            nn.Dropout(dropout),                  
            nn.Linear(self.vae_embed_size, int(self.vae_in_size/2)),
            nn.Sigmoid(),
            nn.Dropout(dropout),                      
            nn.Linear(int(self.vae_in_size/2), self.vae_in_size),
            nn.Sigmoid(),
        )
        
    def get_embed(self, x):
        return self.encode(x)

    def encode(self, x):
        h = self.encode_model(x)
        return h

    def decode(self, z):
        h = self.decode_model(z)
        return h

    def forward(self, x):
        x = self.cast_gpu(x)
        z = self.encode(x.view(-1, self.vae_in_size))
        return self.decode(z)

class VAE(BaseModel):
    def __init__(self, use_gpu):
        super(VAE, self).__init__(use_gpu)
        self.use_gpu = use_gpu
        self.vae_in_size = 392
        self.vae_embed_size =64

        self.encode_model = nn.Sequential(
            nn.Linear(self.vae_in_size, self.vae_in_size//4),
            nn.ReLU(True),    
        )
        self.decode_model = nn.Sequential(
            nn.Linear(self.vae_embed_size, self.vae_in_size//4),
            nn.ReLU(True),
            nn.Linear(self.vae_in_size//4, self.vae_in_size),
        )
        
        
        self.fc21 = nn.Linear(self.vae_in_size//4, self.vae_embed_size)
        self.fc22 = nn.Linear(self.vae_in_size//4, self.vae_embed_size)

            
    def get_embed(self, x):
        mean, _ = self.encode(x)
        return mean

    def encode(self, x):
        h = self.encode_model(x)
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.decode_model(z)
        return torch.sigmoid(h)
        # return h 

    def forward(self, x):
        x = self.cast_gpu(x)
        mu, logvar = self.encode(x.view(-1, self.vae_in_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class Discriminator_SA(BaseModel):
    def __init__(self, use_gpu):
        super(Discriminator_SA, self).__init__(use_gpu)
        dropout = 0.3
        self.input_size = 64
        self.noise_input = 0.01
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, 1)
        )

    def decay_noise(self):
        self.noise_input *= 0.995

    def forward(self, state_action):
        validity = torch.sigmoid(self.model(self.cast_gpu(state_action)))
        validity = torch.clamp(validity, 1e-6, 1-1e-6)
        return validity

"""
gan_v_naive model
"""
# class WoZDiscriminator(BaseModel):
#     def __init__(self, config):
#         super(WoZDiscriminator, self).__init__(config)
#         self.state_in_size = config.vae_embed_size
#         self.action_in_size = 300
#         # self.state_rep = nn.Linear(self.state_in_size * 2, self.state_in_size/2)
#         # self.action_rep = nn.Linear(self.action_in_size * 2, self.action_in_size/2)
#         self.state_rep = nn.Linear(self.state_in_size, self.state_in_size // 2)
#         self.d_rep = nn.Linear(17, 32)
#         self.a_rep = nn.Linear(34, 32)
#         self.s_rep = nn.Linear(82, 32)
#
#         self.noise_input = 0.01
#
#         self.dis_1 = nn.Sequential(
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(config.dropout),
#
#             nn.Linear(self.state_in_size + 32, 32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(config.dropout),
#
#             nn.Linear(32, 1))
#
#         self.dis_2 = nn.Sequential(
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(config.dropout),
#
#             nn.Linear(self.state_in_size + 32, 32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(config.dropout),
#
#             nn.Linear(32, 1))
#
#         self.dis_3 = nn.Sequential(
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(config.dropout),
#
#             nn.Linear(self.state_in_size + 32, 32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(config.dropout),
#
#             nn.Linear(32, 1))
#
#     def decay_noise(self):
#         self.noise_input *= 0.995
#
#     def minibatch_averaging(self, inputs):
#         """
#         This method is explained in the MedGAN paper.
#         """
#         mean_per_feature = torch.mean(inputs, 0)
#         mean_per_feature_repeated = mean_per_feature.repeat(len(inputs), 1)
#         return torch.cat((inputs, mean_per_feature_repeated), 1)
#
#     """
#     here is where it going forward.
#     """
#
#     def forward(self, state_vae, action):
#         """
#         Args:
#             state_vae: [B, 64*3]
#             action: [B, 300]
#         Returns: [B, 3]
#         """
#         # s_z = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, self.noise_input, state.shape))))
#         # s_a = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, self.noise_input, action.shape))))
#         # state_1 = self.state_rep(self.minibatch_averaging(self.cast_gpu(state)))
#         # action_1 = self.action_rep(self.minibatch_averaging(self.cast_gpu(action)))
#
#         emb_onehot = torch.mm(action.squeeze(0), self.mask)
#         emb_domain = emb_onehot[:, :17]
#         emb_action = emb_onehot[:, 17: 17 + 34]
#         emb_slot = emb_onehot[:, 34 + 17:]
#
#         d_emb = self.d_rep(emb_domain)
#         a_emb = self.a_rep(emb_action)
#         s_emb = self.s_rep(emb_slot)
#         # print(state_1.size(), action_1.size())
#         d_vae = state_vae[:, :self.state_in_size]
#         a_vae = state_vae[:, self.state_in_size: self.state_in_size * 2]
#         s_vae = state_vae[:, self.state_in_size * 2:]
#
#         d = torch.cat((d_vae, d_emb), dim=-1)
#         a = torch.cat((a_vae, a_emb), dim=-1)
#         s = torch.cat((s_vae, s_emb), dim=-1)
#
#         d_validity = torch.sigmoid(self.dis_1(d))
#         a_validity = torch.sigmoid(self.dis_2(a))
#         s_validity = torch.sigmoid(self.dis_3(s))
#
#         validity = torch.cat((d_validity, a_validity, s_validity), dim=-1)
#         # using the clamp to make this value looks normal
#         validity = torch.clamp(validity, 1e-7, 1 - 1e-7)
#         # Todo: 1 or 3 over here, I choose 1 this time.
#         validity = torch.mean((validity), dim=-1)
#         return validity
#
#     def forward_wgan(self, state, action):
#         state_1 = self.state_rep(self.cast_gpu(state))
#         action_1 = self.action_rep(self.cast_gpu(action))
#         state_action = torch.cat([state_1, action_1], 1)
#         h = self.model(state_action)
#         return h

"""
gan_v_parallel_curriculum_learning_AE model
"""

# class WoZDiscriminator(BaseModel):
#     def __init__(self, config):
#         super(WoZDiscriminator, self).__init__(config)
#         self.state_in_size = config.vae_embed_size
#         self.action_in_size = 300
#         # self.state_rep = nn.Linear(self.state_in_size * 2, self.state_in_size/2)
#         # self.action_rep = nn.Linear(self.action_in_size * 2, self.action_in_size/2)
#         self.state_rep = nn.Linear(self.state_in_size, self.state_in_size // 2)
#         self.d_rep = nn.Linear(17, 32)
#         self.a_rep = nn.Linear(34, 32)
#         self.s_rep = nn.Linear(82, 32)
#
#         self.noise_input = 0.01
#
#         self.dis_1 = nn.Sequential(
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(config.dropout),
#
#             nn.Linear(self.state_in_size + 32, 32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(config.dropout),
#
#             nn.Linear(32, 1))
#
#         self.dis_2 = nn.Sequential(
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(config.dropout),
#
#             nn.Linear(self.state_in_size + 32, 32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(config.dropout),
#
#             nn.Linear(32, 1))
#
#         self.dis_3 = nn.Sequential(
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(config.dropout),
#
#             nn.Linear(self.state_in_size + 32, 32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(config.dropout),
#
#             nn.Linear(32, 1))
#
#         self.dis_4 = nn.Sequential(
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(config.dropout),
#             nn.Linear(3, 1))
#
#     def decay_noise(self):
#         self.noise_input *= 0.995
#
#     def minibatch_averaging(self, inputs):
#         """
#         This method is explained in the MedGAN paper.
#         """
#         mean_per_feature = torch.mean(inputs, 0)
#         mean_per_feature_repeated = mean_per_feature.repeat(len(inputs), 1)
#         return torch.cat((inputs, mean_per_feature_repeated), 1)
#
#     """
#     here is where it going forward.
#     """
#
#     def forward(self, state_vae, action):
#         """
#         Args:
#             state_vae: [B, 64*3]
#             action: [B, 300]
#         Returns: [B, 3]
#         """
#         # s_z = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, self.noise_input, state.shape))))
#         # s_a = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, self.noise_input, action.shape))))
#         # state_1 = self.state_rep(self.minibatch_averaging(self.cast_gpu(state)))
#         # action_1 = self.action_rep(self.minibatch_averaging(self.cast_gpu(action)))
#
#         emb_onehot = torch.mm(action.squeeze(0), self.mask)
#         emb_domain = emb_onehot[:, :17]
#         emb_action = emb_onehot[:, 17: 17 + 34]
#         emb_slot = emb_onehot[:, 34 + 17:]
#
#         d_emb = self.d_rep(emb_domain)
#         a_emb = self.a_rep(emb_action)
#         s_emb = self.s_rep(emb_slot)
#         # print(state_1.size(), action_1.size())
#         d_vae = state_vae[:, :self.state_in_size]
#         a_vae = state_vae[:, self.state_in_size: self.state_in_size * 2]
#         s_vae = state_vae[:, self.state_in_size * 2:]
#
#         d = torch.cat((d_vae, d_emb), dim=-1)
#         a = torch.cat((a_vae, a_emb), dim=-1)
#         s = torch.cat((s_vae, s_emb), dim=-1)
#
#         """
#         d_h = self.dis_1(d)
#         a_h = self.dis_2(a)
#         s_h = self.dis_3(s)
#         validity_h = torch.cat((d_h, a_h, s_h), dim=-1)
#         validity = torch.sigmoid(self.dis_4(validity_h))
#         # using the clamp to make this value looks normal
#         validity = torch.clamp(validity, 1e-7, 1 - 1e-7)
#         return torch.mean((validity), dim = -1)
#         """
#         # previous code
#         d_validity = torch.sigmoid(self.dis_1(d))
#         a_validity = torch.sigmoid(self.dis_2(a))
#         s_validity = torch.sigmoid(self.dis_3(s))
#         validity = torch.cat((d_validity, a_validity, s_validity), dim=-1)
#         # using the clamp to make this value looks normal
#         validity = torch.clamp(validity, 1e-7, 1 - 1e-7)
#         # validity = torch.mean((validity), dim = -1)
#
#         validity = self.plan(validity, flag = "hard")
#         return validity
#
#     def plan(self, input, flag="mean"):
#         """
#         Args:
#             input: [B, 3]
#             flag:
#         Returns:
#         """
#         if flag == "mean":
#             return torch.mean((input), dim = -1)
#
#         elif flag == "hard":
#             output = self.cast_gpu(tensor([]))
#             # a b c
#             # Todo: how could you set 0.5 over here? Potential bugs over here.
#             for line in input:
#                 if line[0] >= 0.5:
#                     if line[1] >= 0.5:
#                         one = line.unsqueeze(0)
#                     else:
#                         one = self.cast_gpu(tensor([line[0].item(), line[1].item(), 0,])).unsqueeze(0)
#                 else:
#                     one = self.cast_gpu(tensor([line[0].item(), 0., 0,])).unsqueeze(0)
#                 output = torch.cat((output, one), dim = 0)
#             output = torch.mean((output), dim = -1)
#             return output
#
#     def forward_wgan(self, state, action):
#         state_1 = self.state_rep(self.cast_gpu(state))
#         action_1 = self.action_rep(self.cast_gpu(action))
#         state_action = torch.cat([state_1, action_1], 1)
#         h = self.model(state_action)
#         return h

"""
gan_v_parallel_curriculum_learning_VAE model

"""
class WoZDiscriminator(BaseModel):
    def __init__(self, config):
        super(WoZDiscriminator, self).__init__(config)
        self.state_in_size = config.vae_embed_size
        self.action_in_size = 300
        # self.state_rep = nn.Linear(self.state_in_size * 2, self.state_in_size/2)
        # self.action_rep = nn.Linear(self.action_in_size * 2, self.action_in_size/2)
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size // 2)
        self.d_rep = nn.Linear(17, 32)
        self.a_rep = nn.Linear(34, 32)
        self.s_rep = nn.Linear(82, 32)

        self.noise_input = 0.01

        self.dis_1 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            nn.Linear(self.state_in_size + 32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            nn.Linear(32, 1))

        self.dis_2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            nn.Linear(self.state_in_size + 32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            nn.Linear(32, 1))

        self.dis_3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            nn.Linear(self.state_in_size + 32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            nn.Linear(32, 1))

        self.dis_4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(3, 1))

    def decay_noise(self):
        self.noise_input *= 0.995

    def minibatch_averaging(self, inputs):
        """
        This method is explained in the MedGAN paper.
        """
        mean_per_feature = torch.mean(inputs, 0)
        mean_per_feature_repeated = mean_per_feature.repeat(len(inputs), 1)
        return torch.cat((inputs, mean_per_feature_repeated), 1)

    """
    here is where it going forward.
    """

    def forward(self, state_vae, action, surgery):
        """
        Args:
            state_vae: [B, 64*3]
            action: [B, 300]
            surgery: hard / mean / or something like that.
        Returns: [B, 3]
        """
        # s_z = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, self.noise_input, state.shape))))
        # s_a = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, self.noise_input, action.shape))))
        # state_1 = self.state_rep(self.minibatch_averaging(self.cast_gpu(state)))
        # action_1 = self.action_rep(self.minibatch_averaging(self.cast_gpu(action)))

        emb_onehot = torch.mm(action.squeeze(0), self.mask)
        emb_domain = emb_onehot[:, :17]
        emb_action = emb_onehot[:, 17: 17 + 34]
        emb_slot = emb_onehot[:, 34 + 17:]

        d_emb = self.d_rep(emb_domain)
        a_emb = self.a_rep(emb_action)
        s_emb = self.s_rep(emb_slot)
        # print(state_1.size(), action_1.size())
        d_vae = state_vae[:, :self.state_in_size]
        a_vae = state_vae[:, self.state_in_size: self.state_in_size * 2]
        s_vae = state_vae[:, self.state_in_size * 2:]

        d = torch.cat((d_vae, d_emb), dim=-1)
        a = torch.cat((a_vae, a_emb), dim=-1)
        s = torch.cat((s_vae, s_emb), dim=-1)

        d_validity = torch.sigmoid(self.dis_1(d))
        a_validity = torch.sigmoid(self.dis_2(a))
        s_validity = torch.sigmoid(self.dis_3(s))
        validity = torch.cat((d_validity, a_validity, s_validity), dim=-1)
        # using the clamp to make this value looks normal

        validity = self.plan(validity, flag = surgery)
        validity = torch.clamp(validity, 1e-7, 1 - 1e-7)
        return validity

    def plan(self, input, flag="hard"):
        """
        Args:
            input: [B, 3]
            flag:
        Returns:
        """
        if flag == "mean":
            return torch.mean((input), dim = -1)

        elif flag == "hard":
            output = self.cast_gpu(tensor([]))
            # a b c
            # Todo: how could you set 0.5 over here? Potential bugs over here.
            for line in input:
                pass
                if line[0] >= 0.5:
                    if line[1] >= 0.5:
                        one = line.unsqueeze(0)
                    else:
                        one = self.cast_gpu(tensor([line[0].item(), line[1].item(), 0,])).unsqueeze(0)
                else:
                    one = self.cast_gpu(tensor([line[0].item(), 0., 0,])).unsqueeze(0)
                output = torch.cat((output, one), dim = 0)
            output = torch.mean((output), dim = -1)
            return output

        elif flag == "hard_update":
            # a b c
            # Todo: how could you set 0.5 over here? Potential bugs over here.
            # need add some stuff to stabilize the distribution should be around 0.5.

            """
            # archiv, this one is not so useful
            line_1 = input[:,0].unsqueeze(-1)
            line_1_rule  = torch.sigmoid((line_1-0.5)*50)

            line_2 = input[:,1].unsqueeze(-1)
            line_2 = line_1_rule*line_2
            line_2_rule = torch.sigmoid((line_2-0.5)*50)

            line_3 = input[:,0].unsqueeze(-1)
            line_3 = line_2_rule*line_3
            output = line_3 + (line_1+line_2+line_3)/3
            """
            line_1 = input[:,0].unsqueeze(-1)
            line_1_rule  = torch.sigmoid((line_1-0.5)*50)

            line_2 = input[:,1].unsqueeze(-1)
            line_2 = line_1_rule*line_2
            line_2_rule = torch.sigmoid((line_2-0.5)*50)

            line_3 = input[:,0].unsqueeze(-1)
            line_3 = line_2_rule*line_3
            output = (line_1+line_2+line_3)/3
            # output = torch.cat((line_1, line_2, line_3), dim=-1)
            # output = torch.mean((output), dim = -1)
            # output = torch.log(output)-torch.log(1.0-output)
            # output *= 10
            # for i, one in enumerate(output):
                # if one > 0.5:
                #     output[i] = -torch.log(1-one) + np.log(0.5)
                #     # output[i] *= 20
                # else:
                #     output[i] = torch.log(one) - np.log(0.5)
            return output

        elif flag == "soft":
            output = self.cast_gpu(tensor([]))
            for line in input:
                one = self.cast_gpu(tensor([line[0].item(), line[0].item()*line[1].item(), line[0].item()*line[1].item()*line[2].item()])).unsqueeze(0)
                output = torch.cat((output, one), dim = 0)

            output = torch.mean((output), dim = -1)
            output -= 0.4
            output = torch.sigmoid(output*100)

            return output

        elif flag == "hard+soft":
            output = self.cast_gpu(tensor([]))
            # a b c
            # Todo: how could you set 0.5 over here? Potential bugs over here.
            for line in input:
                pass
                if line[0] >= 0.5:
                    if line[1] >= 0.5:
                        one = line.unsqueeze(0)
                    else:
                        one = self.cast_gpu(tensor([line[0].item(), line[1].item(), 0,])).unsqueeze(0)
                else:
                    one = self.cast_gpu(tensor([line[0].item(), 0., 0,])).unsqueeze(0)
                output = torch.cat((output, one), dim = 0)
            output = self.plan(output, flag = "soft")

        elif flag == "product":
            # return the d*s*a
            output = self.cast_gpu(tensor([]))
            for line in input:
                one = line[0]*line[1]*line[2].unsqueeze(0)
                output = torch.cat((output, one), dim = 0)
                # output -= 0.23
                # output = torch.sigmoid(output * 100)
            return output
        elif flag == "sigmoid_prod":
            line_1 = torch.sigmoid((input[:,0]-0.5)*10)
            line_2 = torch.sigmoid((input[:,1]-0.5)*10)
            line_3 = torch.sigmoid((input[:,2]-0.5)*10)

            whole_dis = line_1*line_2*line_3
            output = torch.sigmoid((whole_dis-0.5)*100)
            return output

        elif flag == "d":
            output = input[:,0].unsqueeze(-1)
            # output -= 0.5
            # output = torch.sigmoid(output*10)
            # output = output*(input[:,1].unsqueeze(-1))
            # # times a
            # output -= 0.5
            # output = torch.sigmoid(output*10)
            # output = output*(input[:,2].unsqueeze(-1))
            #
            # output -= 0.5
            # output = torch.sigmoid(output*10)
            return output

        elif flag == "a":
            output = input[:,1].unsqueeze(-1)
            return output

        elif flag == "s":
            output = input[:,2].unsqueeze(-1)
            return output

        elif flag == "da":
            line_1 = input[:,0].unsqueeze(-1)
            line_1_rule  = torch.sigmoid((line_1-0.5)*50)

            line_2 = input[:,1].unsqueeze(-1)
            line_2 = line_1_rule*line_2
            return line_2

        elif flag == "das":
            line_1 = input[:,0].unsqueeze(-1)
            line_1_rule  = torch.sigmoid((line_1-0.5)*20)

            line_2 = input[:,1].unsqueeze(-1)
            line_2 = line_1_rule*line_2
            line_2_rule = torch.sigmoid((line_2-0.5)*20)

            line_3 = input[:,0].unsqueeze(-1)
            line_3 = line_2_rule*line_3
            # Todo: fix the bugs over here.
            # line_3 += (line_1+line_2+line_3)/3
            return line_3

        elif flag == "das_soft":
            line_1 = input[:,0].unsqueeze(-1)
            line_1_rule  = torch.sigmoid((line_1-0.5)*10)

            line_2 = input[:,1].unsqueeze(-1)
            line_2 = line_1_rule*line_2
            line_2_rule = torch.sigmoid((line_2-0.5)*10)

            line_3 = input[:,0].unsqueeze(-1)
            line_3 = line_2_rule*line_3
            # Todo: fix the bugs over here.
            # line_3 += (line_1+line_2+line_3)/3
            return line_3

        elif flag == "min":
            output = torch.min((input), dim = -1)[0]
            return output

    def forward_wgan(self, state, action):
        state_1 = self.state_rep(self.cast_gpu(state))
        action_1 = self.action_rep(self.cast_gpu(action))
        state_action = torch.cat([state_1, action_1], 1)
        h = self.model(state_action)
        return h


class WoZDiscriminator_ziming(BaseModel):
    def __init__(self, config):
        super(WoZDiscriminator_ziming, self).__init__(config)
        dropout = 0.3
        self.state_in_size = 64
        self.action_in_size = 300
        # self.state_rep = nn.Linear(self.state_in_size * 2, self.state_in_size/2)
        # self.action_rep = nn.Linear(self.action_in_size * 2, self.action_in_size/2)
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size // 2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size // 3)

        self.noise_input = 0.01
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(self.state_in_size // 2 + self.action_in_size // 3, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )

    def forward(self, state, action, surgery = ""):
        state_1 = self.state_rep(self.cast_gpu(state))
        action_1 = self.action_rep(self.cast_gpu(action))
        state_action = torch.cat([state_1, action_1], 1)
        validity = torch.sigmoid(self.model(state_action))
        validity = torch.clamp(validity, 1e-7, 1 - 1e-7)
        return validity

class WoZDiscriminator_Update(BaseModel):
    def __init__(self,real_data_feed, use_gpu, batch_size):
        super(WoZDiscriminator_Update, self).__init__(use_gpu)
        dropout = 0.3
        self.batch_size = batch_size
        self.real_data_feed = real_data_feed
        self.loss_BCE = nn.BCELoss()
        self.state_in_size = 64
        self.action_in_size = 300
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size//2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size//3)
        
        self.noise_input = 0.01
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(self.state_in_size//2 + self.action_in_size//3, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )
    def get_optimizer(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=0.0005, betas=(0.5, 0.999))


    def forward(self, state, action):
        state_1 = self.state_rep(self.cast_gpu(state))
        action_1 = self.action_rep(self.cast_gpu(action))
        state_action = torch.cat([state_1, action_1], 1)
        validity = torch.sigmoid(self.model(state_action))
        validity = torch.clamp(validity, 1e-7, 1-1e-7)
        return validity
    
    def sample_real_batch(self):
        batch = self.real_data_feed.next_batch()
        if batch is None:
            self.real_data_feed.epoch_init(shuffle=True, verbose=True)
            batch = self.real_data_feed.next_batch()
        state = batch['state_convlab']
        action = batch['action_id']
        action = one_hot_embedding(action, 300)
        return self.np2var(state, FLOAT), action
    

    def disc_train(self, vae, fake_batch_feed):
        state_rep, action_rep = self.sample_real_batch()
        embed_batch = vae.get_embed(self.cast_gpu(state_rep))
        real_disc_v = self.forward(embed_batch.detach(), self.cast_gpu(action_rep))

        fake_state = cast_type(fake_batch_feed['states'], FLOAT, False)
        fake_state = vae.get_embed(self.cast_gpu(fake_state))
        fake_action =  cast_type(fake_batch_feed['actions'], LONG, False)
        fake_size = len(fake_state)
        fake_action = one_hot_embedding(fake_action, 300)
        
        fake_disc_v = self.forward(fake_state.detach(), fake_action.detach())
        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((self.batch_size,),1.0), torch.full((fake_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        return disc_loss

class WoZDiscriminator_StateVaeActionSeg(WoZDiscriminator):
    def __init__(self, use_gpu):
        super(WoZDiscriminator_StateVaeActionSeg, self).__init__(use_gpu)
        self.state_in_size = 64
        self.action_in_size = 160
        dropout = 0.3
        # self.state_rep = nn.Linear(self.state_in_size * 2, self.state_in_size/2)
        # self.action_rep = nn.Linear(self.action_in_size * 2, self.action_in_size/2)
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size//2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size//3)
        
        self.noise_input = 0.01
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(self.state_in_size//2 + self.action_in_size//3, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

class A2C_Discriminator(BaseModel):
    def __init__(self, config, use_gpu, real_data_feed, batch_size):
        super(A2C_Discriminator, self).__init__(config)
        self.real_data_feed = real_data_feed
        self.batch_size = batch_size
        dropout = 0.3
        self.loss_BCE = nn.BCELoss()
        self.state_in_size = 392
        self.action_in_size = 300
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size//2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size//3)

        self.model = nn.Sequential(
            nn.Linear(self.state_in_size + self.action_in_size, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    
    def get_optimizer(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=0.0005, betas=(0.5, 0.999))

    def sample_real_batch(self):
        batch = self.real_data_feed.next_batch()
        if batch is None:
            self.real_data_feed.epoch_init(shuffle=True, verbose=True)
            batch = self.real_data_feed.next_batch()
        state = batch['state_convlab']
        action = batch['action_id']
        action = one_hot_embedding(action, 300)
        return self.np2var(state, FLOAT), action

    def sample_real_batch_id(self):
        batch = self.real_data_feed.next_batch()
        if batch is None:
            self.real_data_feed.epoch_init(shuffle=True, verbose=True)
            batch = self.real_data_feed.next_batch()
        state = batch['state_convlab']
        action = batch['action_id']
        return self.np2var(state, FLOAT), self.np2var(action, INT)


    def forward(self, state, action):
        # state_1 = self.state_rep(self.cast_gpu(state))
        # action_1 = self.action_rep(self.cast_gpu(action))
        state_action = torch.cat([state, action], 1)
        validity = torch.sigmoid(self.model(state_action))
        validity = torch.clamp(validity, 1e-7, 1-1e-7)
        return validity

    def get_reward(self, batch_feed):
        state = cast_type(batch_feed['states'], FLOAT, False)
        action = cast_type(batch_feed['actions'], LONG, False)
        action = one_hot_embedding(action, 300)
        fake_disc_v = self.forward(state.detach(), action.detach())
        return fake_disc_v.detach().view(-1)
      
    def disc_train(self, fake_batch_feed):
        # batch
        real_state, real_action= self.sample_real_batch()
        real_disc_v = self.forward(real_state, real_action)

        fake_state = cast_type(fake_batch_feed['states'], FLOAT, False)
        fake_action =  cast_type(fake_batch_feed['actions'], LONG, False)
        fake_size = len(fake_state)
        fake_action = one_hot_embedding(fake_action, 300)
        # print(len(real_state), len(fake_state))
        # assert len(real_state)==len(fake_state)

        fake_disc_v = self.forward(fake_state.detach(), fake_action.detach())
        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((self.batch_size,),1.0), torch.full((fake_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        return disc_loss

class AIRL(BaseModel):
    def __init__(self, config, use_gpu, real_data_feed, batch_size):
        super(AIRL, self).__init__(config)
        self.real_data_feed = real_data_feed
        self.batch_size = batch_size
        dropout = 0.3
        self.loss_BCE = nn.BCELoss()
        self.state_in_size = 392
        self.action_in_size = 300
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size//2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size//3)

        self.model_g = nn.Sequential(
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.state_in_size + self.action_in_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.model_h = nn.Sequential(
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.state_in_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    
    def get_optimizer(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=0.0005, betas=(0.5, 0.999))

    def sample_real_batch(self):
        batch = self.real_data_feed.next_batch()
        if batch is None:
            self.real_data_feed.epoch_init(shuffle=True, verbose=True)
            batch = self.real_data_feed.next_batch()
        state = batch['state_convlab']
        action = batch['action_id']
        action = one_hot_embedding(action, 300)
        next_state = batch['state_convlab_next']
        return self.np2var(state, FLOAT), action, self.np2var(next_state, FLOAT)
    
    def sample_real_batch_id(self):
        batch = self.real_data_feed.next_batch()
        if batch is None:
            self.real_data_feed.epoch_init(shuffle=True, verbose=True)
            batch = self.real_data_feed.next_batch()
        state = batch['state_convlab']
        action = batch['action_id']
        return self.np2var(state, FLOAT), self.np2var(action, INT)


    def forward(self, state, action, state_next):
        # state_1 = self.state_rep(self.cast_gpu(state))
        # action_1 = self.action_rep(self.cast_gpu(action))
        # state_2 = self.state_rep(self.cast_gpu(state_next))

        state_1 = self.cast_gpu(state)
        action_1 = self.cast_gpu(action)
        state_2 = self.cast_gpu(state_next)

        state_action = torch.cat([state_1, action_1], -1)
        validity = self.model_g(state_action) + 0.99 * self.model_h(state_2) - self.model_h(state_1)
        return validity

    def get_reward(self, batch_feed):
        state = cast_type(batch_feed['states'], FLOAT, False)
        action = cast_type(batch_feed['actions'], LONG, False)
        action = one_hot_embedding(action, 300)
        state_next = cast_type(batch_feed['next_states'], FLOAT, False)
        fake_disc_v = self.forward(state.detach(), action.detach(), state_next.detach())
        return fake_disc_v.detach().view(-1)
      
    def disc_train(self, fake_batch_feed):
        # batch
        real_state, real_action, real_state_next= self.sample_real_batch()
        real_disc_v = self.forward(real_state, real_action, real_state_next)

        fake_state = cast_type(fake_batch_feed['states'], FLOAT, False)
        fake_action =  cast_type(fake_batch_feed['actions'], LONG, False)
        fake_state_next = cast_type(fake_batch_feed['next_states'], FLOAT, False)
        fake_size = len(fake_state)
        fake_action = one_hot_embedding(fake_action, 300)

        fake_disc_v = self.forward(fake_state.detach(), fake_action.detach(), fake_state_next.detach())
        loss = - real_disc_v.mean() + fake_disc_v.mean()
        return loss

class AE_3(BaseModel):
    def __init__(self, config):
        super(AE_3, self).__init__(config)
        self.config = config
        self.vae_in_size = config.state_out_size
        self.input_size = config.state_out_size
        self.embedding_size = config.vae_embed_size
        self.temperature = config.gumbel_temp

        # Encoder(Embedding)_part
        self.e_1 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, self.embedding_size),
            nn.LeakyReLU())

        self.e_2 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, self.embedding_size),
            nn.LeakyReLU())

        self.e_3 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, self.embedding_size),
            nn.LeakyReLU())

        # classification
        self.c_1 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 17))

        self.c_2 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 34))

        self.c_3 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 82))

        # Todo, change to 64*3 -> 64*4
        self.c_whole = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * 3, self.embedding_size*3),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * 3, 300))

        self.decoder = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size*3, 256),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, self.input_size),
            nn.LeakyReLU())

        # output Layer
        self.softmax = nn.Softmax(dim=-1)
        self.gumbel_connector = GumbelConnector(self.use_gpu)


    def get_embed(self, x):
        """
        Args:
            x: states
        Returns: d, a, s
        """
        d = self.e_1(x)
        a = self.e_2(x)
        s = self.e_3(x)

        state_emb = torch.cat((d, a, s), dim = -1)
        return state_emb

    def get_pred(self, z_1, z_2, z_3):
        """
        Args:
            z_1: random noise  [B, 64]
            z_2: random noise  coming from G, after learning/ from real case
            z_3: random noise  [B, 64]
        Returns:
        """
        z = torch.cat((z_1, z_2, z_3), dim = -1)
        action_h = self.c_whole(z)
        action_pred = self.gumbel_connector(action_h, temperature = self.temperature, hard = True)
        return action_pred

    def forward(self, state, action):
        """
        Args:
            state:
            action:

        Returns: Decoding and Encoding bf states.
        """
        e_1 = self.e_1(state)
        e_2 = self.e_2(state)
        e_3 = self.e_3(state)

        e = torch.cat((e_1, e_2, e_3), dim = -1)
        recon_batch = self.decoder(e)
        return recon_batch

    def forward_add_4(self, state, action):
        """
        Args:
            state: [B, 392]
            action: [B, 300]
        Returns: 4 actions and one reconstruction belief states
        """
        e_1 = self.e_1(state)
        e_2 = self.e_2(state)
        e_3 = self.e_3(state)

        e = torch.cat((e_1, e_2, e_3), dim = -1)

        # the sigmoid function will add in loss function.
        recon_batch = self.decoder(e)
        domain = self.softmax(self.c_1(e_1))
        action = self.softmax(self.c_2(e_2))
        slot  =  self.softmax(self.c_3(e_3))
        # should add gumber not softmax.
        action_whole = self.c_whole(e)
        action_whole = self.gumbel_connector(action_whole, temperature = self.temperature, hard = False)

        return recon_batch, domain, action, slot, action_whole

    def classfier(self, state, action):
        pass

class AE_3_parallel_VAE(BaseModel):
    def __init__(self, config):
        super(AE_3_parallel_VAE, self).__init__(config)
        self.config = config
        self.vae_in_size = config.state_out_size
        self.input_size = config.state_out_size
        self.embedding_size = config.vae_embed_size
        self.temperature = config.gumbel_temp

        # 392 -> 256
        # self.e_common = nn.Sequential(
        #     nn.Dropout(config.dropout),
        #     nn.Linear(self.input_size, 256),
        #     nn.LeakyReLU())
        """
        # (Embedding)_part
        """
        self.e_1 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, self.embedding_size),
            nn.LeakyReLU())

        self.e_2 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, self.embedding_size),
            nn.LeakyReLU())

        self.e_3 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, self.embedding_size),
            nn.LeakyReLU())

        """
        VAE part
        """
        self.hidden2mean_1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.hidden2logv_1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.latent2hidden_1=nn.Linear(self.embedding_size, self.embedding_size)

        self.hidden2mean_2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.hidden2logv_2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.latent2hidden_2=nn.Linear(self.embedding_size, self.embedding_size)

        self.hidden2mean_3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.hidden2logv_3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.latent2hidden_3=nn.Linear(self.embedding_size, self.embedding_size)
        """
        classification, no relu after the network.
        """
        self.c_1 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, self.embedding_size),

            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 17))

        self.c_2 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, self.embedding_size),

            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 34))

        self.c_3 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, self.embedding_size),

            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 82))

        """
        last one of classification
        """
        # 192 -> 300
        self.c_action = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * 3, 256),

            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 300))

        """
        decoder part
        # 192 -> 392
        # no activation after this one, since add to sigmoid in BCE with logits.        
        """
        self.decoder = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * 3, 256),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 392))
        # output Layer
        self.softmax = nn.Softmax(dim=-1)
        self.gumbel_connector = GumbelConnector(config.use_gpu)

    def get_embed(self, x):
        """
        Args:
            x: states
        Returns: d, a, s
        """
        # [B, 392] -> [B, 256]

        batch_size = x.size(0)
        # common_h = self.e_common(state)
        # [B, 64] -> [B, seq_len, 64] = [B, 3, 64]
        d_e = self.e_1(x)
        a_e = self.e_2(x)
        s_e = self.e_3(x)
        # [B, 3, 64]

        # get the vae seperately
        mean_1 = self.hidden2mean_1(d_e)
        logv_1 = self.hidden2logv_1(d_e)
        z_1 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        std_1 = torch.exp(0.5 * logv_1)
        z_1 = z_1 * std_1 + mean_1

        mean_2 = self.hidden2mean_2(a_e)
        logv_2 = self.hidden2logv_2(a_e)
        z_2 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        std_2 = torch.exp(0.5 * logv_2)
        z_2 = z_2 * std_2 + mean_2

        mean_3 = self.hidden2mean_3(s_e)
        logv_3 = self.hidden2logv_3(s_e)
        z_3 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        std_3 = torch.exp(0.5 * logv_3)
        z_3 = z_3 * std_3 + mean_3

        state_emb = torch.cat((z_1, z_2, z_3), dim = -1)

        # return all of this stuff.
        return state_emb

    def get_pred(self, z_1, z_2, z_3):
        """
        Args:
            z_1: random noise  [B, 64]
            z_2: random noise  coming from G, after learning/ from real case
            z_3: random noise  [B, 64]
        Returns:
        """
        z = torch.cat((z_1, z_2, z_3), dim = -1)
        action_h = self.c_action(z)
        action_pred = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)
        return action_pred

    def get_fake_data(self, z_1, z_2, z_3):
        """
        Args:
            z_1:
            z_2:
            z_3:
        Returns: action
        """
        raise NotImplementedError
        # the fake action should comes from z, but we only use mean in embedding for real states.
        fake_input = torch.cat((z_1, z_2, z_3), dim = -1)

        action_h = self.c_action(fake_input)
        fake_actions = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)

        return fake_actions

    def forward(self, state, action):
        """
        Args:
            state:
            action:
        Returns: Decoding and Encoding bf states.
        """
        batch_size = state.size(0)
        # common_h = self.e_common(state)
        # [B, 64] -> [B, seq_len, 64] = [B, 3, 64]
        d_e = self.e_1(state)
        a_e = self.e_2(state)
        s_e = self.e_3(state)
        # [B, 3, 64]

        # First classification
        # domain_pred = self.softmax(self.c_1(d_e))
        # action_pred = self.softmax(self.c_2(a_e))
        # slot_pred  =  self.softmax(self.c_3(s_e))
        # get the whole embedding [domain, action, slot]
        # get the vae seperately
        mean_1 = self.hidden2mean_1(d_e)
        logv_1 = self.hidden2logv_1(d_e)
        z_1 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        std_1 = torch.exp(0.5 * logv_1)
        z_1 = z_1 * std_1 + mean_1

        mean_2 = self.hidden2mean_2(a_e)
        logv_2 = self.hidden2logv_2(a_e)
        z_2 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        std_2 = torch.exp(0.5 * logv_2)
        z_2 = z_2 * std_2 + mean_2

        mean_3 = self.hidden2mean_3(s_e)
        logv_3 = self.hidden2logv_3(s_e)
        z_3 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        std_3 = torch.exp(0.5 * logv_3)
        z_3 = z_3 * std_3 + mean_3

        # classification should over here.
        domain_pred = self.softmax(self.c_1(z_1))
        action_pred = self.softmax(self.c_2(z_2))
        slot_pred  =  self.softmax(self.c_3(z_3))

        d_vae_e = self.latent2hidden_1(z_1)
        a_vae_e = self.latent2hidden_2(z_2)
        s_vae_e = self.latent2hidden_3(z_3)

        e_vae = torch.cat((d_vae_e, a_vae_e, s_vae_e), dim = -1)
        action_h = self.c_action(e_vae)
        action_pred_whole = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)
        recon_batch = self.decoder(e_vae)
        # return all of this stuff.
        return recon_batch, domain_pred, action_pred, slot_pred, action_pred_whole

# class AE_3_parallel_VAE(BaseModel):
#     def __init__(self, config):
#         super(AE_3_parallel_VAE, self).__init__(config)
#         self.config = config
#         self.vae_in_size = config.state_out_size
#         self.input_size = config.state_out_size
#         self.embedding_size = config.vae_embed_size
#         self.temperature = config.gumbel_temp
#
#         # 392 -> 256
#         self.e_common = nn.Sequential(
#             nn.Dropout(config.dropout),
#             nn.Linear(self.input_size, 256),
#             nn.LeakyReLU())
#         """
#         # (Embedding)_part
#         """
#         self.e_1 = nn.Sequential(
#             nn.Dropout(config.dropout),
#             nn.Linear(self.input_size, 128),
#             nn.LeakyReLU(),
#             nn.Dropout(config.dropout),
#             nn.Linear(128, self.embedding_size),
#             nn.LeakyReLU())
#
#         self.e_2 = nn.Sequential(
#             nn.Dropout(config.dropout),
#             nn.Linear(self.input_size, 128),
#             nn.LeakyReLU(),
#             nn.Dropout(config.dropout),
#             nn.Linear(128, self.embedding_size),
#             nn.LeakyReLU())
#
#         self.e_3 = nn.Sequential(
#             nn.Dropout(config.dropout),
#             nn.Linear(self.input_size, 128),
#             nn.LeakyReLU(),
#             nn.Dropout(config.dropout),
#             nn.Linear(128, self.embedding_size),
#             nn.LeakyReLU())
#         # self.e_1 = nn.Sequential(
#         #     nn.Dropout(config.dropout),
#         #     nn.Linear(256, 128),
#         #     nn.LeakyReLU(),
#         #     nn.Dropout(config.dropout),
#         #     nn.Linear(128, self.embedding_size),
#         #     nn.LeakyReLU())
#         #
#         # self.e_2 = nn.Sequential(
#         #     nn.Dropout(config.dropout),
#         #     nn.Linear(256, 128),
#         #     nn.LeakyReLU(),
#         #     nn.Dropout(config.dropout),
#         #     nn.Linear(128, self.embedding_size),
#         #     nn.LeakyReLU())
#         #
#         # self.e_3 = nn.Sequential(
#         #     nn.Dropout(config.dropout),
#         #     nn.Linear(256, 128),
#         #     nn.LeakyReLU(),
#         #     nn.Dropout(config.dropout),
#         #     nn.Linear(128, self.embedding_size),
#         #     nn.LeakyReLU())
#
#         """
#         VAE part
#         """
#         self.hidden2mean_1 = nn.Linear(self.embedding_size, self.embedding_size)
#         self.hidden2logv_1 = nn.Linear(self.embedding_size, self.embedding_size)
#         self.latent2hidden_1=nn.Linear(self.embedding_size, self.embedding_size)
#
#         self.hidden2mean_2 = nn.Linear(self.embedding_size, self.embedding_size)
#         self.hidden2logv_2 = nn.Linear(self.embedding_size, self.embedding_size)
#         self.latent2hidden_2=nn.Linear(self.embedding_size, self.embedding_size)
#
#         self.hidden2mean_3 = nn.Linear(self.embedding_size, self.embedding_size)
#         self.hidden2logv_3 = nn.Linear(self.embedding_size, self.embedding_size)
#         self.latent2hidden_3=nn.Linear(self.embedding_size, self.embedding_size)
#
#
#         """
#         classification, no relu after the network.
#         """
#         self.c_1 = nn.Sequential(
#             nn.Dropout(config.dropout),
#             nn.Linear(self.embedding_size, self.embedding_size),
#
#             nn.LeakyReLU(),
#             nn.Dropout(config.dropout),
#             nn.Linear(self.embedding_size, 17))
#
#         self.c_2 = nn.Sequential(
#             nn.Dropout(config.dropout),
#             nn.Linear(self.embedding_size, self.embedding_size),
#
#             nn.LeakyReLU(),
#             nn.Dropout(config.dropout),
#             nn.Linear(self.embedding_size, 34))
#
#         self.c_3 = nn.Sequential(
#             nn.Dropout(config.dropout),
#             nn.Linear(self.embedding_size, self.embedding_size),
#
#             nn.LeakyReLU(),
#             nn.Dropout(config.dropout),
#             nn.Linear(self.embedding_size, 82))
#
#         """
#         last one of classification
#         """
#         # 192 -> 300
#         self.c_action = nn.Sequential(
#             nn.Dropout(config.dropout),
#             nn.Linear(self.embedding_size * 3, 256),
#
#             nn.LeakyReLU(),
#             nn.Dropout(config.dropout),
#             nn.Linear(256, 300))
#
#         """
#         decoder part
#         # 192 -> 392
#         # no activation after this one, since add to sigmoid in BCE with logits.
#         """
#         self.decoder = nn.Sequential(
#             nn.Dropout(config.dropout),
#             nn.Linear(self.embedding_size * 3, 256),
#             nn.LeakyReLU(),
#             nn.Dropout(config.dropout),
#             nn.Linear(256, 392))
#         # output Layer
#         self.softmax = nn.Softmax(dim=-1)
#         self.gumbel_connector = GumbelConnector(config.use_gpu)
#
#     def get_embed(self, x):
#         """
#         Args:
#             x: states
#         Returns: d, a, s
#         """
#         # [B, 392] -> [B, 256]
#         # Todo: chage this one to the mean.
#         batch_size = x.size(0)
#         # common_h = self.e_common(state)
#         # [B, 64] -> [B, seq_len, 64] = [B, 3, 64]
#         d_e = self.e_1(x)
#         a_e = self.e_2(x)
#         s_e = self.e_3(x)
#         # [B, 3, 64]
#
#         # get the vae seperately
#         mean_1 = self.hidden2mean_1(d_e)
#         logv_1 = self.hidden2logv_1(d_e)
#         z_1 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
#         std_1 = torch.exp(0.5 * logv_1)
#         z_1 = z_1 * std_1 + mean_1
#
#         mean_2 = self.hidden2mean_2(a_e)
#         logv_2 = self.hidden2logv_2(a_e)
#         z_2 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
#         std_2 = torch.exp(0.5 * logv_2)
#         z_2 = z_2 * std_2 + mean_2
#
#         mean_3 = self.hidden2mean_3(s_e)
#         logv_3 = self.hidden2logv_3(s_e)
#         z_3 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
#         std_3 = torch.exp(0.5 * logv_3)
#         z_3 = z_3 * std_3 + mean_3
#
#         state_emb = torch.cat((mean_1, mean_2, mean_3), dim = -1)
#
#         # return all of this stuff.
#         return state_emb
#
#     def get_pred(self, z_1, z_2, z_3):
#         """
#         Args:
#             z_1: random noise  [B, 64]
#             z_2: random noise  coming from G, after learning/ from real case
#             z_3: random noise  [B, 64]
#         Returns:
#         """
#         z = torch.cat((z_1, z_2, z_3), dim = -1)
#         action_h = self.c_action(z)
#         action_pred = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)
#         return action_pred
#
#     def get_fake_data(self, z_1, z_2, z_3):
#         """
#         Args:
#             z_1:
#             z_2:
#             z_3:
#         Returns: action
#         """
#         raise NotImplementedError
#         # the fake action should comes from z, but we only use mean in embedding for real states.
#         fake_input = torch.cat((z_1, z_2, z_3), dim = -1)
#
#         action_h = self.c_action(fake_input)
#         fake_actions = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)
#
#         return fake_actions
#
#     def forward(self, state, action):
#         """
#         Args:
#             state:
#             action:
#         Returns: Decoding and Encoding bf states.
#         """
#         batch_size = state.size(0)
#         # [B, 64] -> [B, seq_len, 64] = [B, 3, 64]
#         d_e = self.e_1(state)
#         a_e = self.e_2(state)
#         s_e = self.e_3(state)
#         # [B, 3, 64]
#
#         # First classification
#         # domain_pred = self.softmax(self.c_1(d_e))
#         # action_pred = self.softmax(self.c_2(a_e))
#         # slot_pred  =  self.softmax(self.c_3(s_e))
#         # get the whole embedding [domain, action, slot]
#         # get the vae seperately
#         mean_1 = self.hidden2mean_1(d_e)
#         logv_1 = self.hidden2logv_1(d_e)
#         z_1 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
#         std_1 = torch.exp(0.5 * logv_1)
#         z_1 = z_1 * std_1 + mean_1
#
#         mean_2 = self.hidden2mean_2(a_e)
#         logv_2 = self.hidden2logv_2(a_e)
#         z_2 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
#         std_2 = torch.exp(0.5 * logv_2)
#         z_2 = z_2 * std_2 + mean_2
#
#         mean_3 = self.hidden2mean_3(s_e)
#         logv_3 = self.hidden2logv_3(s_e)
#         z_3 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
#         std_3 = torch.exp(0.5 * logv_3)
#         z_3 = z_3 * std_3 + mean_3
#
#         # classification should over here.
#         domain_pred = self.softmax(self.c_1(z_1))
#         action_pred = self.softmax(self.c_2(z_2))
#         slot_pred  =  self.softmax(self.c_3(z_3))
#
#         d_vae_e = self.latent2hidden_1(z_1)
#         a_vae_e = self.latent2hidden_2(z_2)
#         s_vae_e = self.latent2hidden_3(z_3)
#
#         e_vae = torch.cat((d_vae_e, a_vae_e, s_vae_e), dim = -1)
#         action_h = self.c_action(e_vae)
#         action_pred_whole = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)
#         recon_batch = self.decoder(e_vae)
#         # return all of this stuff.
#         # get KL_loss
#         d_kl_loss = (- 0.5 * torch.sum(1 + logv_1 - mean_1.pow(2) - logv_1.exp()))
#         a_kl_loss = (- 0.5 * torch.sum(1 + logv_2 - mean_2.pow(2) - logv_2.exp()))
#         s_kl_loss = (- 0.5 * torch.sum(1 + logv_3 - mean_3.pow(2) - logv_3.exp()))
#         # Todo, chang this one to the seperate training.
#         ELBO_whole = d_kl_loss + a_kl_loss + s_kl_loss
#         return recon_batch, domain_pred, action_pred, slot_pred, action_pred_whole, ELBO_whole


class AE_3_parallel_VAE_finish(BaseModel):
    def __init__(self, config):
        super(AE_3_parallel_VAE_finish, self).__init__(config)
        self.config = config
        self.vae_in_size = config.state_out_size
        self.input_size = config.state_out_size
        self.embedding_size = config.vae_embed_size
        self.temperature = config.gumbel_temp

        # 392 -> 256
        self.e_common = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_size, 256),
            nn.LeakyReLU())
        """
        # (Embedding)_part
        """
        self.e_1 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, self.embedding_size),
            nn.LeakyReLU())

        self.e_2 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, self.embedding_size),
            nn.LeakyReLU())

        self.e_3 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, self.embedding_size),
            nn.LeakyReLU())

        """
        Reparamater part
        """
        self.hidden2mean_1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.hidden2logv_1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.latent2hidden_1=nn.Linear(self.embedding_size, self.embedding_size)

        self.hidden2mean_2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.hidden2logv_2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.latent2hidden_2=nn.Linear(self.embedding_size, self.embedding_size)

        self.hidden2mean_3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.hidden2logv_3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.latent2hidden_3=nn.Linear(self.embedding_size, self.embedding_size)


        """
        classification, no relu after the network.
        """
        self.c_1 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, self.embedding_size),

            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 17))

        self.c_2 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, self.embedding_size),

            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 34))

        self.c_3 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, self.embedding_size),

            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 82))

        """
        decoder part
        first do the reparamater sample trick, and using Z_whole to decode.
        # 192 -> 392
        # no activation after this one, since add to sigmoid in BCE with logits.        
        """
        self.z2mean = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * 3, self.embedding_size * 3))

        self.z2var = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * 3, self.embedding_size * 3))

        self.decoder = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * 3, 256),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 392))
        # output Layer
        self.softmax = nn.Softmax(dim=-1)
        self.gumbel_connector = GumbelConnector(config.use_gpu)

    def get_embed(self, x):
        """
        Args:
            x: states
        Returns: d, a, s
        """
        # [B, 392] -> [B, 256]
        # Todo: chage this one to the mean.
        batch_size = x.size(0)
        # common_h = self.e_common(state)
        # [B, 64] -> [B, seq_len, 64] = [B, 3, 64]
        d_e = self.e_1(x)
        a_e = self.e_2(x)
        s_e = self.e_3(x)
        # [B, 3, 64]

        # get the vae seperately
        mean_1 = self.hidden2mean_1(d_e)
        logv_1 = self.hidden2logv_1(d_e)
        z_1 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        std_1 = torch.exp(0.5 * logv_1)
        z_1 = z_1 * std_1 + mean_1

        mean_2 = self.hidden2mean_2(a_e)
        logv_2 = self.hidden2logv_2(a_e)
        z_2 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        std_2 = torch.exp(0.5 * logv_2)
        z_2 = z_2 * std_2 + mean_2

        mean_3 = self.hidden2mean_3(s_e)
        logv_3 = self.hidden2logv_3(s_e)
        z_3 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        std_3 = torch.exp(0.5 * logv_3)
        z_3 = z_3 * std_3 + mean_3

        # state_emb = torch.cat((mean_1, mean_2, mean_3), dim = -1)
        # state_emb = torch.cat((mean_1, mean_2, mean_3), dim = -1)
        state_emb = torch.cat((z_1, z_2, z_3), dim = -1)

        # return all of this stuff.
        return state_emb

    def get_pred(self, z_1, z_2, z_3):
        """
        Args:
            z_1: random noise  [B, 64]
            z_2: random noise  coming from G, after learning/ from real case
            z_3: random noise  [B, 64]
        Returns:
        """
        z = torch.cat((z_1, z_2, z_3), dim = -1)
        action_h = self.c_action(z)
        action_pred = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)
        return action_pred

    def get_fake_data(self, z_1, z_2, z_3):
        """
        Args:
            z_1:
            z_2:
            z_3:
        Returns: action
        """
        raise NotImplementedError
        # the fake action should comes from z, but we only use mean in embedding for real states.
        fake_input = torch.cat((z_1, z_2, z_3), dim = -1)

        action_h = self.c_action(fake_input)
        fake_actions = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)

        return fake_actions

    def forward(self, state, action):
        """
        Args:
            state:
            action:
        Returns: Decoding and Encoding bf states.
        """
        batch_size = state.size(0)
        # [B, 64] -> [B, seq_len, 64] = [B, 3, 64]
        d_e = self.e_1(state)
        a_e = self.e_2(state)
        s_e = self.e_3(state)
        # [B, 3, 64]

        # First classification
        # domain_pred = self.softmax(self.c_1(d_e))
        # action_pred = self.softmax(self.c_2(a_e))
        # slot_pred  =  self.softmax(self.c_3(s_e))
        # get the whole embedding [domain, action, slot]
        # get the vae seperately
        mean_1 = self.hidden2mean_1(d_e)
        logv_1 = self.hidden2logv_1(d_e)
        z_1 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        std_1 = torch.exp(0.5 * logv_1)
        z_1 = z_1 * std_1 + mean_1

        mean_2 = self.hidden2mean_2(a_e)
        logv_2 = self.hidden2logv_2(a_e)
        z_2 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        std_2 = torch.exp(0.5 * logv_2)
        z_2 = z_2 * std_2 + mean_2

        mean_3 = self.hidden2mean_3(s_e)
        logv_3 = self.hidden2logv_3(s_e)
        z_3 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        std_3 = torch.exp(0.5 * logv_3)
        z_3 = z_3 * std_3 + mean_3

        # classification should over here.
        domain_pred = self.softmax(self.c_1(z_1))
        action_pred = self.softmax(self.c_2(z_2))
        slot_pred  =  self.softmax(self.c_3(z_3))

        # d_vae_e = self.latent2hidden_1(z_1)
        # a_vae_e = self.latent2hidden_2(z_2)
        # s_vae_e = self.latent2hidden_3(z_3)
        z_pre = torch.cat((z_1, z_2, z_3), dim = -1)
        mean_4 = self.z2mean(z_pre)
        logv_4 = self.z2var(z_pre)
        z_dec = self.cast_gpu(torch.randn([batch_size, self.embedding_size*3]))
        z_dec = z_dec*logv_4 + mean_4

        recon_batch = self.decoder(z_dec)
        # return all of this stuff.
        # get KL_loss
        d_kl_loss = (- 0.5 * torch.sum(1 + logv_1 - mean_1.pow(2) - logv_1.exp()))
        a_kl_loss = (- 0.5 * torch.sum(1 + logv_2 - mean_2.pow(2) - logv_2.exp()))
        s_kl_loss = (- 0.5 * torch.sum(1 + logv_3 - mean_3.pow(2) - logv_3.exp()))
        z_last_kl_loss = (- 0.5 * torch.sum(1 + logv_4 - mean_4.pow(2) - logv_4.exp()))

        # Todo, chang this one to the seperate training.
        ELBO_whole = d_kl_loss + a_kl_loss + s_kl_loss + z_last_kl_loss
        return recon_batch, domain_pred, action_pred, slot_pred, ELBO_whole
