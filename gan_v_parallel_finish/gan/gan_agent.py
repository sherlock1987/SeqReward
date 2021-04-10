# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import os
from collections import defaultdict
import logging
from torch.autograd import Variable
import torch.nn as nn
import gan_model_sat, gan_model_vae
from gan_model import Discriminator, Generator, ContEncoder
from gan_model_vae import VAE, AutoEncoder, VAE_StateActionEmbed
from torch_utils import one_hot_embedding, LookupProb
from laed.dataset.corpora import PAD, EOS, EOT, BOS
from laed.utils import Pack, INT, FLOAT, LONG, cast_type
from utils import BCELoss_double, cal_accuracy
import torch.nn.functional as F
import random
import pickle

logger = logging.getLogger()
"""
Agent of every GAN, include D, G, VAE inside.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

class GanRnnAgent(nn.Module):
    def __init__(self, corpus, config, action2name):
        super(GanRnnAgent, self).__init__()
        self.use_gpu = config.use_gpu
        
        if config.state_type=='rnn':
            self.vocab = corpus.vocab
            self.rev_vocab = corpus.rev_vocab
            self.vocab_size = len(self.vocab)
            self.go_id = self.rev_vocab[BOS]
            self.eos_id = self.rev_vocab[EOS]
            self.context_encoder = ContEncoder(corpus, config)
            
        self.action2name=action2name
        self.lookupProb_ = LookupProb(action2name, config)
        self.discriminator = Discriminator(config)
        self.generator = Generator(config)

        self.loss_BCE = nn.BCELoss(reduction = "sum")
        self.config = config

    def cast_gpu(self, var):
        if self.config.use_gpu:
            return var.cuda().float()
        else:
            return var
        
    def binary2onehot(self,x):
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
    
    def shuffle_action(self, x):
        # len(x[0]) == 18   
        m_new = []
        for col in range(0, 18, 2):
            if np.random.random()>0.5:
                m_new.append(x[:,col])
                m_new.append(x[:,col+1])
            else:
                m_new.append(x[:,col+1])
                m_new.append(x[:,col])
        return torch.stack(m_new).transpose(1,0)

    
            

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        if type(inputs)==list:
            return cast_type(Variable(torch.Tensor(inputs)), dtype,
                         self.use_gpu)
        return cast_type(Variable(torch.from_numpy(inputs)), dtype,
                         self.use_gpu)

    def sample_step(self, sample_shape):
        batch_size, state_noise_dim, action_noise_dim = sample_shape
        z_noise = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, state_noise_dim)))))
        # a_z = Variable(torch.Tensor(np.random.normal(0, 1, (batch_size,  action_noise_dim))))
        # sample onhot tensor to represent the sampled actions
        # a_z = self.cast_gpu(Variable(torch.Tensor(np.eye(self.config.action_num)[np.random.choice(self.config.action_num, batch_size)])))
        state_rep, action_rep = self.generator(z_noise, z_noise)
        # print(torch.max(action_rep.data, dim=1))
        # return state_rep, z_noise
        return state_rep, action_rep


    def gen_train(self, sample_shape):
        state_rep, action_rep =self.sample_step(sample_shape)
        mean_dist_1 = - (state_rep.mean(dim=0) - state_rep).pow(2).mean() * 0.00003
        mean_dist_2 = - (action_rep.mean(dim=0) - action_rep).pow(2).mean() * 0.00003
        mean_dist = mean_dist_1 + mean_dist_2
        disc_v = self.discriminator(state_rep, action_rep)
        gen_loss = -torch.mean(torch.log(disc_v))
        gen_loss = Pack(gen_loss= gen_loss, mean_dist = mean_dist)
        return gen_loss, (state_rep.detach(), action_rep.detach())
    
    def gen_validate(self, sample_shape, record_gen=False):
        state_rep, action_rep =self.sample_step(sample_shape)
        # policy_prob = self.policy_validate(state_rep, action_rep)
        if record_gen:
            return [], [state_rep.detach().cpu().tolist(), action_rep.detach().cpu().tolist()]
        else:
            return [], []
    
    
    def policy_validate_for_human(self, sample_shape, batch_feed):
        action_data_feed = one_hot_embedding(batch_feed['action_id'], self.config.action_num)
        batch_size = sample_shape[0]
        real_state_rep = self.context_encoder(batch_feed).detach()
        policy_prob = self.policy_validate(real_state_rep, action_data_feed)
        return policy_prob.detach()
    
    def self_shuffle_disc_train(self, sample_shape, batch_feed):
        batch_size = sample_shape[0]
        real_state_rep, action_data_feed = self.read_real_data(sample_shape, batch_feed)
        fake_state_rep=real_state_rep[:,torch.randperm(real_state_rep.size()[1])]
        real_disc_v = self.discriminator(real_state_rep, action_data_feed)
        fake_disc_v = self.discriminator(fake_state_rep.detach(), action_data_feed.detach())

        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((batch_size,),1.0), torch.full((batch_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        # disc_loss = BCELoss_double(rf_disc_v.view(-1), labels)
        # disc_loss = - torch.mean(real_disc_v) + torch.mean(fake_disc_v)
        disc_loss = Pack(disc_loss=disc_loss)
        return disc_loss, (fake_state_rep.detach(), action_data_feed.detach())


    def disc_train(self, sample_shape, batch_feed, fake_state_action=None):
        batch_size = sample_shape[0]
        real_state_rep, action_data_feed = self.read_real_data(sample_shape, batch_feed)
        real_disc_v = self.discriminator(real_state_rep, action_data_feed)
        if fake_state_action is None:
            fake_state_rep, fake_action_rep = self.sample_step(sample_shape)
        else:
            fake_state_rep, fake_action_rep = fake_state_action
        # if self.config.round_for_disc:
        #     fake_disc_v = self.discriminator(fake_state_rep.detach().round(), fake_action_rep.detach().round())
        # else:            
        if np.random.random()<0.5:
            fake_state_rep, fake_action_rep = real_state_rep.detach(), action_data_feed[:,torch.randperm(action_data_feed.size()[1])] 
        fake_disc_v = self.discriminator(fake_state_rep.detach(), fake_action_rep.detach())

        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((batch_size,),1.0), torch.full((batch_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        # disc_loss = BCELoss_double(rf_disc_v.view(-1), labels)
        # disc_loss = - torch.mean(real_disc_v) + torch.mean(fake_disc_v)
        disc_loss = Pack(disc_loss=disc_loss)
        disc_acc = cal_accuracy(rf_disc_v.view(-1), labels_one)
        return disc_loss, disc_acc

    def policy_validate(self, state, action):
        sum_prob = []
        fc1_out =self.p_fc1(state)
        py_logits = self.p_y(torch.tanh(fc1_out)).view(-1, self.config.k)
        log_py = F.log_softmax(py_logits, dim=py_logits.dim()-1)
        #log_py = F.softmax(py_logits, dim=py_logits.dim()-1)
        log_py = log_py.view(-1, self.config.y_size, self.config.k)
        for log_py_line, action_line in zip(log_py, action):
            prob_line = self.lookupProb_(log_py_line, action_line)
            if type(prob_line)!=int and torch.isnan(prob_line):
                print(state[:2])
            sum_prob.append(prob_line)
        return torch.mean(torch.FloatTensor(sum_prob))

    def read_real_data(self, sample_shape, batch_feed):
        action_data_feed = one_hot_embedding(batch_feed['action_id'], self.config.action_num)
        if self.config.state_type=='rnn':
            real_state_rep = self.context_encoder(batch_feed).detach()
        elif self.config.state_type=='table':
            real_state_rep = self.np2var(batch_feed['state_table'], FLOAT)
        return real_state_rep, action_data_feed

class WGanAgent(GanRnnAgent):
    def __init__(self, corpus, config, action2name):
        super(WGanAgent, self).__init__(corpus, config, action2name)
    def gen_train(self, sample_shape):
        state_rep, action_rep =self.sample_step(sample_shape)
        disc_v = self.discriminator.forward_wgan(state_rep, action_rep)
        gen_loss = -torch.mean(disc_v)
        gen_loss = Pack(gen_loss= gen_loss)
        return gen_loss, (state_rep, action_rep)

    def disc_train(self, sample_shape, batch_feed, fake_state_action=None):
        batch_size = sample_shape[0]
        real_state_rep, action_data_feed = self.read_real_data(sample_shape, batch_feed)
        real_disc_v = self.discriminator.forward_wgan(real_state_rep, action_data_feed)
        if fake_state_action is None:
            fake_state_rep, fake_action_rep = self.sample_step(sample_shape)
        else:
            fake_state_rep, fake_action_rep = fake_state_action
        fake_disc_v = self.discriminator.forward_wgan(fake_state_rep.detach(), fake_action_rep.detach())

        real_disc_loss = - torch.mean(real_disc_v) 
        fake_disc_loss = torch.mean(fake_disc_v)
        disc_loss = real_disc_loss + fake_disc_loss
        disc_loss = Pack(disc_loss=disc_loss)
        disc_acc = np.array([-real_disc_loss.item(), fake_disc_loss.item()])
        return disc_loss, disc_acc

class GanAgent_AutoEncoder(GanRnnAgent):
    def __init__(self, corpus, config, action2name):
        super(GanAgent_AutoEncoder, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_vae.Discriminator(config)
        self.generator = gan_model_vae.Generator(config)
        self.state_out_size = config.state_out_size
        self.vae = gan_model_vae.AutoEncoder(config)
        # self.vae_in_size = config.state_out_size + config.action_num
        self.autoencoder_in_size = config.state_out_size + 9
        self.config = config 
        
        
    def vae_train(self, batch_feed):
        state_rep, action_rep = self.read_real_data(None, batch_feed)
        state_action = torch.cat([self.cast_gpu(state_rep), self.cast_gpu(action_rep)], -1)
        recon_batch = self.vae(state_action)
        loss = self.autoencoder_loss(recon_batch, state_action)
        vae_loss = Pack(vae_loss=loss) 
        return vae_loss
        
        
    def read_real_data(self, sample_shape, batch_feed):
        action_data_feed = self.np2var(batch_feed['action_id_binary'], FLOAT).view(-1, 9)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed
    
    def autoencoder_loss(self, recon_x, x):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size))
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).mean()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        l2_loss = self.vae.l2_norm()
        loss = BCE + l2_loss
        return loss + l2_loss
    
    
    def sample_step(self, sample_shape):
        batch_size, state_noise_dim, action_noise_dim = sample_shape
        z_noise = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, state_noise_dim)))))
        state_action_rep = self.generator(z_noise, z_noise)
        discrete_state_action_rep = self.get_decode_result(state_action_rep)
        # state_rep, action_rep = torch.split(discrete_state_action_rep, 392, dim=1)       
        state_rep, action_rep = discrete_state_action_rep[:,:self.state_out_size].clone(), discrete_state_action_rep[:,self.state_out_size:].clone()      
        return state_rep, action_rep

    def get_decode_result(self, state_action_rep):
        result = self.vae.decode(state_action_rep)
        return result
    
    def gan_vae_optimizer(self, config):
        params = list(self.generator.parameters()) + list(self.vae.parameters())
        if config.op == 'adam':
            print("Use adam")
            return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           params), lr=config.init_lr, betas=(0.5, 0.999))
        elif config.op == 'sgd':
            print("Use SGD")
            return torch.optim.SGD(params, lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            print("RMSProp")
            return torch.optim.RMSprop(params, lr=config.init_lr,
                                       momentum=config.momentum)

class GanAgent_AutoEncoder_Encode(GanAgent_AutoEncoder):
    def __init__(self, corpus, config, action2name):
        super(GanAgent_AutoEncoder_Encode, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_vae.Discriminator(config)
        self.autoencoder_in_size = config.state_out_size + 300
    def vae_train(self, batch_feed):
        state_rep, action_rep = self.read_real_data_onehot(None, batch_feed)
        state_action = torch.cat([self.cast_gpu(state_rep), self.cast_gpu(action_rep)], -1)
        recon_batch = self.vae(state_action)
        loss = self.autoencoder_loss(recon_batch, state_action)
        vae_loss = Pack(vae_loss=loss) 
        return vae_loss
        
        
    def read_real_data(self, sample_shape, batch_feed):
        action_data_feed = self.np2var(batch_feed['action_id_binary'], FLOAT).view(-1, 9)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed

    def read_real_data_onehot(self, sample_shape, batch_feed):
        action_data_feed = one_hot_embedding(batch_feed['action_id'], 300)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed
    
    def autoencoder_loss(self, recon_x, x):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size))
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).mean()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        l2_loss = self.vae.l2_norm()
        loss = BCE + l2_loss
        return loss + l2_loss
    
    
    def sample_step(self, sample_shape):
        batch_size, state_noise_dim, action_noise_dim = sample_shape
        z_noise = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, state_noise_dim)))))
        state_action_rep = self.generator(z_noise, z_noise)
        return state_action_rep

    def get_vae_embed(self, state_action_rep):
        result = self.vae.encode(state_action_rep)
        return result
    
    def gan_vae_optimizer(self, config):
        params = list(self.generator.parameters()) + list(self.vae.parameters())
        if config.op == 'adam':
            print("Use adam")
            return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           params), lr=config.init_lr, betas=(0.5, 0.999))
        elif config.op == 'sgd':
            print("Use SGD")
            return torch.optim.SGD(params, lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            print("RMSProp")
            return torch.optim.RMSprop(params, lr=config.init_lr,
                                       momentum=config.momentum)
            
    def gen_train(self, sample_shape):
        state_action_rep =self.sample_step(sample_shape)
        mean_dist = - (state_action_rep.mean(dim=0) - state_action_rep).pow(2).sum() * 0.00003
        disc_v = self.discriminator(state_action_rep)
        gen_loss = -torch.mean(torch.log(disc_v))
        gen_loss = Pack(gen_loss= gen_loss, mean_dist = mean_dist)
        return gen_loss, state_action_rep.detach()
        
    
    def disc_train(self, sample_shape, batch_feed, fake_state_action=None):
        batch_size = sample_shape[0]
        # state_rep, action_rep = self.read_real_data(None, batch_feed)
        state_rep, action_rep = self.read_real_data_onehot(None, batch_feed)
        state_action = torch.cat([self.cast_gpu(state_rep), self.cast_gpu(action_rep)], -1)
        embed_batch = self.vae.encode(state_action)
        real_disc_v = self.discriminator(embed_batch.detach())
        if fake_state_action is None:
            fake_state_action = self.sample_step(sample_shape)
        else:
            fake_state_action = fake_state_action
        if self.config.round_for_disc:
            fake_disc_v = self.discriminator(fake_state_action.detach())
        else:            
            fake_disc_v = self.discriminator(fake_state_action.detach())

        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((batch_size,),1.0), torch.full((batch_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        # disc_loss = BCELoss_double(rf_disc_v.view(-1), labels)
        # disc_loss = - torch.mean(real_disc_v) + torch.mean(fake_disc_v)
        disc_loss = Pack(disc_loss=disc_loss)
        disc_acc = cal_accuracy(rf_disc_v.view(-1), labels_one)
        return disc_loss, disc_acc

    def gen_validate(self, sample_shape, record_gen=False):
        state_action_rep =self.sample_step(sample_shape)
        # policy_prob = self.policy_validate(state_rep, action_rep)
        if record_gen:
            return [], state_action_rep
        else:
            return [], []

class GanAgent_AutoEncoder_State(GanAgent_AutoEncoder):
    # only state is fed to autoencoder
    def __init__(self, corpus, config, action2name):
        super(GanAgent_AutoEncoder_State, self).__init__(corpus, config, action2name)
        self.autoencoder_in_size = config.state_out_size
        """
        Specify the model you use over here.
        """
        if config.vae: self.vae = gan_model_vae.VAE_3(config)
        else: self.vae = gan_model_vae.AE_3(config)
        # else: self.vae = gan_model_vae.AE_3_parallel_AE(config)
        # else: self.vae = gan_model_vae.AE_3_parallel_VAE(config)
        # else: self.vae = gan_model_vae.AE_3_parallel_VAE_finish(config)
        self.discriminator = gan_model_sat.WoZDiscriminator(config)
        self.generator = gan_model_sat.WoZGenerator_StateVae(config)
        # self.generator = gan_model_sat.WoZGenerator_StateVae_finish(config)

    def vae_train(self, batch_feed):
        # state_rep, action_rep = self.read_real_data_onehot(None, batch_feed)
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        state_rep = self.cast_gpu(state_rep)
        recon_batch = self.vae(state_rep)
        loss = self.autoencoder_loss(recon_batch, state_rep)
        vae_loss = Pack(vae_loss=loss) 
        return vae_loss

    def read_real_data(self, sample_shape, batch_feed):
        action_data_feed = self.np2var(batch_feed['action_id_binary'], FLOAT).view(-1, 9)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed

    def read_real_data_onehot(self, sample_shape, batch_feed):
        action_id = self.binary2onehot(batch_feed['action_id_binary'])
        action_data_feed = self.np2var(action_id, FLOAT).view(-1, 18)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed

    def read_real_data_onehot_300(self, sample_shape, batch_feed):
        action_data_feed = one_hot_embedding(batch_feed['action_id'], 300)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed
    
    def autoencoder_loss(self, recon_x, x):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size))
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).mean()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        l2_loss = self.vae.l2_norm()
        loss = BCE + l2_loss
        return loss + l2_loss

    def generate_fake_plus(self, real_states, real_action, fake_states, training_mode = "000", change = False):
        """
        Args:
            real_states: [B, 192]
            fake_states: [B, 192]
            training_mode: 001/110
            change: bool
        Returns: [B, 192]
        """
        if training_mode == ["0", "0", "0"]:
            if np.random.random() < 0.5 and change == True:
                state_rep_f, action_rep_f = real_states.detach(), real_action[:, torch.randperm(real_action.size(1))]
                return state_rep_f, action_rep_f
            else:
                d_emb = fake_states[:, :64]
                a_emb = fake_states[:, 64:128]
                s_emb = fake_states[:, 128:]

                state_rep_f, action_rep_f = self.vae.get_fake_data(d_emb, a_emb, s_emb)
                return state_rep_f, action_rep_f
        else:
            for idx, ele in enumerate(training_mode):
                if idx == 0:
                    if ele == "0":
                        d_emb = fake_states[:, :64]
                    else:
                        d_emb = real_states[:, :64]
                elif idx == 1:
                    if ele == "0":
                        a_emb = fake_states[:, 64:128]
                    else:
                        a_emb = real_states[:, 64:128]
                else:
                    if ele == "0":
                        s_emb = fake_states[:, 128:]
                    else:
                        s_emb = real_states[:, 128:]
        state_rep_f, action_rep_f = self.vae.get_fake_data(d_emb, a_emb, s_emb)
        return state_rep_f, action_rep_f

    def sample_step_selection(self, sample_shape, real_data, training_mode = "000", change = False):
        # This is for D training.
        """
        Args:
            sample_shape:
            real_data: tuple
            training_mode: 110/001, this meaning only training the third D or First two D.
            change: bool for changing 0.5 prob.
        Returns:
        """
        batch_size, state_noise_dim, action_noise_dim = sample_shape
        real_states, real_action = real_data[0], real_data[1]
        training_mode = [training_mode[0], training_mode[1], training_mode[2]]
        z_noise = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, state_noise_dim)))))
        # for 001 training
        if np.random.random() < 0.5 and change == True:
            state_rep_f, action_rep_f = real_states.detach(), real_action[:, torch.randperm(real_action.size(1))]
            return state_rep_f, action_rep_f
        else:
            fake_states, fake_action = self.generator(z_noise)
            state_rep_f = fake_states
            action_rep_f = fake_action
            # action_rep_f = self.vae.get_pred(state_rep_f[:, :64], state_rep_f[:, 64:128], state_rep_f[:, 128:])
            # state_rep_f , action_rep_f = self.vae.get_fake_data(fake_states[:, :64], fake_states[:, 64:128], fake_states[:, 128:])
            return state_rep_f, action_rep_f

    def sample_step(self, sample_shape):
        # useless in new code
        """
        Args:
            sample_shape: [B, 128(state_noise_dim), 128(action_noise_dim_nouse)]
        Returns:
        """
        batch_size, state_noise_dim, action_noise_dim = sample_shape
        z_noise = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, state_noise_dim)))))
        # Only have one noise this time.
        # state_emb, action_pred = self.generator(z_noise)
        state_emb, action_pred = self.generator(z_noise)
        # action_pred = self.vae.get_pred(state_emb[:, :64], state_emb[:, 64:128], state_emb[:, 128:])
        return state_emb, action_pred

    def get_vae_embed(self, state_action_rep):
        result = self.vae.encode(state_action_rep)
        return result
    
    def gan_vae_optimizer(self, config):
        params = list(self.generator.parameters()) + list(self.vae.parameters())
        if config.op == 'adam':
            print("Use adam")
            return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            params), lr=config.init_lr, betas=(0.5, 0.999))
        elif config.op == 'sgd':
            print("Use SGD")
            return torch.optim.SGD(params, lr=config.init_lr,
                                    momentum=config.momentum)
        elif config.op == 'rmsprop':
            print("RMSProp")
            return torch.optim.RMSprop(params, lr=config.init_lr,
                                        momentum=config.momentum)
            
    def gen_train(self, batch_feed, sample_shape, training_mode = "000"):
        # use the real data 0_noise, 1_real, 1_real
        """
        Args:
            batch_feed: training set for auxilary learning
            sample_shape: [B, 128, 128]
            training_mode: 001/100
        Returns:
        """
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        embed_batch = self.vae.get_embed(self.cast_gpu(state_rep))
        state_rep_f, action_rep_f = self.sample_step_selection(sample_shape, real_data=(embed_batch, action_rep), training_mode=training_mode, change=True)
        mean_dist_1 = - (state_rep.mean(dim=0) - state_rep).pow(2).mean() * self.config.sim_factor
        mean_dist_2 = - (action_rep.mean(dim=0) - action_rep).pow(2).mean() * self.config.sim_factor
        mean_dist = mean_dist_1 + mean_dist_2
        # mean_dist =  mean_dist_2  1/0
        disc_v = self.discriminator(state_rep_f, action_rep_f)

        # get socre seperately, for all of them.
        # training_mode = "000"
        training_mode = [training_mode[0], training_mode[1], training_mode[2]]
        gen_loss = 0.
        for idx, ele in enumerate(training_mode):
            if ele == "0":
                gen_loss += torch.mean(torch.log(disc_v[:, idx]))

        gen_loss = Pack(gen_loss= gen_loss, mean_dist = mean_dist)
        return gen_loss, (state_rep_f.detach(), action_rep_f.detach())

    """
    1. get results from real data.
    2. get results from fake data.    all the fake data is coming from the G.
    3. [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0] and then compute BCE loss. That's it. Meaning he training the model in a group way.
    """
    def disc_train(self, sample_shape, batch_feed, D_criterions, training_mode = "000", fake_state_action=None):
        """
        Args:
            sample_shape: [B, 128, 128]
            batch_feed:
            D_criterions: [[ ], [ ], [ ]]
            training_mode: 0 means fake data, 1 means real data.
            fake_state_action: (d_emb, a_emb, s_emb), action_pred
        Returns:
        """
        batch_size = sample_shape[0]
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        """
        Real data = [state, action] -> 3 * [state_emb, action_divide]
        """
        embed_batch = self.vae.get_embed(self.cast_gpu(state_rep))
        real_disc_v = self.discriminator(embed_batch.detach(), self.cast_gpu(action_rep))

        """
        Fake data = [gen_states, gen_action] or [gen_states, original_action_shuffle]
        """
        if fake_state_action is None:
            state_rep_f, action_rep_f = self.sample_step_selection(sample_shape, real_data=(embed_batch, action_rep), training_mode = training_mode, change=True)
        else:

            # This is for training on history data.
            state_rep_f, action_rep_f = fake_state_action
        # if np.random.random()<0.5:
        #     state_rep_f, action_rep_f = embed_batch.detach(), action_rep[:,torch.randperm(action_rep.size()[1])]
        # [B, 3]
        # test_action = action_rep[0,:]
        test_sum = torch.sum(action_rep)
        if test_sum != batch_size:
            raise ValueError
        fake_disc_v = self.discriminator(state_rep_f.detach(), action_rep_f.detach())
        # fake_disc_v = self.discriminator(embed_batch.detach(), self.cast_gpu(action_rep))
        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim = 0)
        # get loss next:
        # label has some problem, right?
        # half_labels = torch.tensor([])
        # for ele in training_mode:
        #     if ele == "0":
        #         one = torch.full((batch_size, 1), 1.0)
        #     else:
        #         one = torch.full((batch_size, 1), 0.0)
        #     half_labels = torch.cat((half_labels, one), dim = -1)


        labels_one = torch.cat([torch.full((batch_size, 3), 1.0), torch.full((batch_size, 3), 0.0)], dim = 0)
        # labels_one = torch.cat([torch.full((batch_size, 3), 1.0), half_labels], dim = 0)

        labels_one = self.cast_gpu(labels_one)
        # Todo, fix the label stuff.
        disc_loss, output_selection, labels_selection = self.get_d_loss(D_criterions, rf_disc_v, labels_one, training_mode)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        # disc_loss = BCELoss_double(rf_disc_v.view(-1), labels)
        # disc_loss = - torch.mean(real_disc_v) + torch.mean(fake_disc_v)
        disc_acc = cal_accuracy(output_selection.view(-1), labels_selection.view(-1))
        return disc_loss, disc_acc

    def gen_validate(self, sample_shape, record_gen=False):
        state_rep, action_rep =self.sample_step(sample_shape)
        # policy_prob = self.policy_validate(state_rep, action_rep)
        if record_gen:
            return state_rep, action_rep
        else:
            return [], []

    def get_d_loss(self, D_criterions, output, labels, training_mode = "000"):
        """
        get loss seperately
        Args:
            D_criterions:
            output: output of two D :[2×B, 3]
            labels: [2*B, 3]
            training_mode: 001/110
        Returns: loss(PACK), output_selection, labels_selection
        """
        training_mode = [training_mode[0], training_mode[1], training_mode[2]]
        output_tensor = tensor([])
        labels_tensor = tensor([])
        loss = 0.
        for idx, ele in enumerate(training_mode):
            if idx == 0:
                if ele == "0":
                    output_1 = output[:, 0].unsqueeze(-1)
                    label_1 = labels[:, 0].unsqueeze(-1)
                    loss += D_criterions[0](output_1, label_1)
                    output_tensor = torch.cat((output_tensor, output_1), dim = -1)
                    labels_tensor = torch.cat((labels_tensor, label_1), dim = -1)

            elif idx == 1:
                if ele == "0":
                    output_2 = output[:, 1].unsqueeze(-1)
                    label_2 = labels[:, 1].unsqueeze(-1)
                    loss += D_criterions[1](output_2, label_2)
                    output_tensor = torch.cat((output_tensor, output_2), dim = -1)
                    labels_tensor = torch.cat((labels_tensor, label_2), dim = -1)

            elif idx == 2:
                if ele == "0":
                    output_3 = output[:, 2].unsqueeze(-1)
                    label_3 = labels[:, 2].unsqueeze(-1)
                    loss += D_criterions[2](output_3, label_3)
                    output_tensor = torch.cat((output_tensor, output_3), dim = -1)
                    labels_tensor = torch.cat((labels_tensor, label_3), dim = -1)
        return Pack(disc_loss = loss), output_tensor, labels_tensor
    
class GanAgent_VAE_State(GanAgent_AutoEncoder_State):
    # only state is fed to VAE, action is onehot with 300 dims
    # This one is only for VAE training.
    def __init__(self, corpus, config, action2name):
        super(GanAgent_VAE_State, self).__init__(corpus, config, action2name)
        self.autoencoder_in_size = config.state_out_size
        self.config = config

        # read mask file
        with open(config.mask_onehot_path, 'rb') as f:
            self.mask = pickle.load(f).to(device).float()
            # self.domain_dic, self.action_dic, self.slot_dic = self._get_label()

        if config.vae: self.vae = gan_model_vae.VAE_3(config)
        else: self.vae = gan_model_vae.AE_3(config)
        # else: self.vae = gan_model_vae.AE_3_parrallel(config)
        # else: self.vae = gan_model_vae.AE_3_parallel_AE(config)
        # else: self.vae = gan_model_vae.AE_3_parallel_VAE_finish(config)
        # self.discriminator =gan_model_sat.WoZDiscriminator(config)
        # self.generator =gan_model_sat.WoZGenerator_StateVae_finish(config)

    def vae_train(self, batch_feed, epoch):
        """
        Args:
            batch_feed:
            epoch:
        Returns:
        """
        # state_rep, action_rep = self.read_real_data_onehot(None, batch_feed)
        state, action = self.read_real_data_onehot_300(None, batch_feed)
        state = self.cast_gpu(state)
        action = self.cast_gpu(action)
        if not self.config.vae:
            # for parallel_VAE
            recon_batch, d_pred_1, a_pred_1, s_pred_1, ELBO_whole = self.vae(state, action)
            # This will compute only one thound percent KL_loss.
            fix_weight = 0.0
            kl_weight = self.kl_anneal_function(epoch+1, 1, 600, 1200)
            vae_loss, d_loss_1, a_loss_1, s_loss_1 = self.ae_loss_parallel_3(self.cast_gpu(recon_batch), state, d_pred_1, a_pred_1, s_pred_1, action)
            class_weight = 0.1
            loss = Pack(vae_loss = vae_loss, class_loss = (d_loss_1+ a_loss_1 + s_loss_1), ELBO_whole = fix_weight * kl_weight * ELBO_whole)

        else:
            # follows the paper, but this one has some bugs.
            # Todo, add the prediction if you needed, but I don't think you should add that.
            # Todo, seems like you fail on this one, but just don't give up, okay? code is your life, be nice to it.
            # VAE part, I define three VAE using three latent variables.
            # self.get_label(action)
            (mean_1, logvar_1, z_1), (mean_2, logvar_2, z_2), (mean_3, logvar_3, z_3), (domain_pred, action_pred, slot_pred) = self.vae.encoder(state, action)
            label_d, label_a, label_s = self.get_label(action)
            d_grouped_mu, d_grouped_logvar, d_list, d_list_num = self.accumulate_group_evidence(mean_1, logvar_1, label_d)
            a_grouped_mu, a_grouped_logvar, a_list, a_list_num = self.accumulate_group_evidence(mean_2, logvar_2, label_a)
            s_grouped_mu, s_grouped_logvar, s_list, s_list_num = self.accumulate_group_evidence(mean_3, logvar_3, label_s)

            d_z, d_indexes, d_sizes = self.group_wise_reparameterize_each(training=True, mu=d_grouped_mu, logvar=d_grouped_logvar, labels_batch=label_d,
                list_groups_labels = d_list, sizes_group=d_list_num)

            a_z, a_indexes, a_sizes = self.group_wise_reparameterize_each(training=True, mu=a_grouped_mu, logvar=a_grouped_logvar, labels_batch=label_a,
                list_groups_labels = a_list, sizes_group=a_list_num)

            s_z, s_indexes, s_sizes = self.group_wise_reparameterize_each(training=True, mu=s_grouped_mu, logvar=s_grouped_logvar, labels_batch=label_s,
                list_groups_labels = s_list, sizes_group=s_list_num)

            # KL divergence for domain_group, using positive, and useing the reconstruction - KL loss
            domain_kl_loss = -0.5 * (- 1 - d_grouped_logvar + d_grouped_mu.pow(2) + d_grouped_logvar.exp()).sum()

            # KL divergence for action_group
            action_kl_loss = -0.5 * (- 1 - a_grouped_logvar + a_grouped_mu.pow(2) + a_grouped_logvar.exp()).sum()

            # KL divergence for slot, no group currently
            slot_kl_loss = -0.5 * (- 1 - s_grouped_logvar + s_grouped_mu.pow(2) + s_grouped_logvar.exp()).sum()

            recon_batch = self.vae.decoder(d_z, a_z, s_z)
            recon_loss = self.vae_3_loss(self.cast_gpu(recon_batch), state)
            ELBO = (recon_loss + domain_kl_loss + action_kl_loss + slot_kl_loss)/state.size(0)
            # loss = Pack(vae_loss=loss, d_kl_loss = domain_kl_loss, a_kl_loss = action_kl_loss, s_kl_loss = slot_kl_loss)
            loss = Pack(vae_loss = ELBO)
        return loss

    def vae_validation(self, batch_feed):
        state, action = self.read_real_data_onehot_300(None, batch_feed)
        state = self.cast_gpu(state)
        action = self.cast_gpu(action)
        if not self.config.vae:
            # for parallel VAE
            recon_batch, d_pred_1, a_pred_1, s_pred_1, ELBO_whole = self.vae(state, action)
            vae_loss = torch.sum(torch.abs((torch.sigmoid(recon_batch) > 0.5).float() - state)).to("cuda")

            emb_onehot = torch.mm(action.squeeze(0), self.mask).to(device)
            emb_domain = emb_onehot[:, :17]
            emb_action = emb_onehot[:, 17: 17+34]
            emb_slot = emb_onehot[:, 34+17:]
            # compute onehot evaluation of this stuff.
            d_loss = self.eval_hard(d_pred_1, emb_domain)
            a_loss = self.eval_hard(a_pred_1, emb_action)
            s_loss = self.eval_hard(s_pred_1, emb_slot)

            loss = Pack(vae_loss = vae_loss, d_loss = d_loss, a_loss = a_loss, s_loss = s_loss, ELBO_whole = ELBO_whole)

            """
            # for single stuff.
            recon_batch_d, recon_batch_a, recon_batch_s, d_pred_1, a_pred_1, s_pred_1, d_kl_loss, a_kl_loss, s_kl_loss = self.vae(state, action)
            vae_loss = torch.sum(torch.abs((torch.sigmoid(recon_batch_d) > 0.5).float() - state)).to("cuda")

            emb_onehot = torch.mm(action.squeeze(0), self.mask).to(device)
            emb_domain = emb_onehot[:, :17]
            emb_action = emb_onehot[:, 17: 17+34]
            emb_slot = emb_onehot[:, 34+17:]
            # compute onehot evaluation of this stuff.
            d_loss = self.eval_hard(d_pred_1, emb_domain)

            loss = Pack(vae_loss = vae_loss, d_loss = d_loss, ELBO_whole = d_kl_loss)
            """

        else:
            # VAE part
            # Todo, add the prediction if you needed, but I don't think you should add that.
            # VAE part, I define three VAE using three latent variables.
            (mean_1, logvar_1, z_1), (mean_2, logvar_2, z_2), (mean_3, logvar_3, z_3), (domain_pred, action_pred, slot_pred) = self.vae.encoder(state, action)
            label_d, label_a, label_s = self.get_label(action)
            d_grouped_mu, d_grouped_logvar, d_list, d_list_num = self.accumulate_group_evidence(mean_1, logvar_1, label_d)
            a_grouped_mu, a_grouped_logvar, a_list, a_list_num = self.accumulate_group_evidence(mean_2, logvar_2, label_a)
            s_grouped_mu, s_grouped_logvar, s_list, s_list_num = self.accumulate_group_evidence(mean_3, logvar_3, label_s)

            d_z, d_indexes, d_sizes = self.group_wise_reparameterize_each(training=True, mu=d_grouped_mu, logvar=d_grouped_logvar, labels_batch=label_d,
                list_groups_labels = d_list, sizes_group=d_list_num)

            a_z, a_indexes, a_sizes = self.group_wise_reparameterize_each(training=True, mu=a_grouped_mu, logvar=a_grouped_logvar, labels_batch=label_a,
                list_groups_labels = a_list, sizes_group=a_list_num)

            s_z, s_indexes, s_sizes = self.group_wise_reparameterize_each(training=True, mu=s_grouped_mu, logvar=s_grouped_logvar, labels_batch=label_s,
                list_groups_labels = s_list, sizes_group=s_list_num)

            # KL divergence for domain_group, using positive, and useing the reconstruction - KL loss
            domain_kl_loss = -(- 0.5 * torch.sum(1 + d_grouped_logvar - d_grouped_mu.pow(2) - d_grouped_logvar.exp()))

            # KL divergence for action_group
            action_kl_loss = -(- 0.5 * torch.sum(1 + a_grouped_logvar - a_grouped_mu.pow(2) - a_grouped_logvar.exp()))

            # KL divergence for slot, no group currently
            slot_kl_loss = -(- 0.5 * torch.sum(1 + s_grouped_logvar - s_grouped_mu.pow(2) - s_grouped_logvar.exp()))

            recon_batch = self.vae.decoder(d_z, a_z, s_z)
            # ELBO = (recon_loss - domain_kl_loss - action_kl_loss - slot_kl_loss)/state.size(0)
            vae_loss = torch.sum(torch.abs((torch.sigmoid(recon_batch) > 0.5).float() - state)).to("cuda")
            loss = Pack(vae_loss=vae_loss, d_loss = -domain_kl_loss, a_loss = -action_kl_loss, s_loss = -slot_kl_loss)
            # loss = Pack(vae_loss = loss)
        return loss

    def vae_loss(self, recon_x, x, mu, logvar):
        if self.config.vae_loss=='bce':
            pos_weights = torch.full([392], 15).to(device)
            Reconstruction_loss = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weights)
            BCE = Reconstruction_loss(recon_x, x.view(-1, self.autoencoder_in_size))


        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l2_loss = self.vae.l2_norm()
        loss = (BCE + KLD)/len(x.view(-1))
        return loss + l2_loss

    """
    Two way of computing loss function. Forward AE_3 or combine with the classification problem.
    """
    def ae_loss(self, recon_x, x):
        if self.config.vae_loss == 'bce':
            pos_weights = torch.full([392], 15).to(device)
            Reconstruction_loss = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weights)
            BCE = Reconstruction_loss(recon_x, x.view(-1, self.autoencoder_in_size))
        elif self.config.vae_loss == 'mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).sum()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        loss = BCE/len(x.view(-1))
        return loss

    def ae_loss_3(self, recon_x, x, d_pred, a_pred, s_pred, action_pred, action):
        """
        Args:
            recon_x: reconstructed bf
            x: original x
            action: action for computing classification loss
        Returns: loss
        """
        if self.config.vae_loss == 'bce':
            pos_weights = torch.full([392], 15).to(device)
            Reconstruction_loss = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weights)
            bce_loss = Reconstruction_loss(recon_x, x.view(-1, self.autoencoder_in_size))

            # compute other loss
            # [B, 300] * [300, size]
            classification_loss = torch.nn.BCELoss(reduction="sum")

            emb_onehot = torch.mm(action.squeeze(0), self.mask)
            emb_domain = emb_onehot[:, :17]
            emb_action = emb_onehot[:, 17: 17+34]
            emb_slot = emb_onehot[:, 34+17:]

            d_loss = classification_loss(d_pred, emb_domain)
            a_loss = classification_loss(a_pred, emb_action)
            s_loss = classification_loss(s_pred, emb_slot)
            action_loss = classification_loss(action_pred, action)
            return bce_loss, d_loss, a_loss, s_loss, action_loss

        elif self.config.vae_loss == 'mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).sum()
            raise ValueError("you did not implement this one in this function!!")
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
    """
    parallel loss with seperate loss for domain, action and slot.
    """
    def ae_loss_parallel_3(self, recon_x, x, d_pred_1, a_pred_1, s_pred_1, action):
        """
        Args:
            recon_x: reconstructed bf
            x: original x
            action: action for computing classification loss
            d, a, s
        Returns: loss
        """
        if self.config.vae_loss == 'bce':
            pos_weights = torch.full([392], 15).to(device)
            Reconstruction_loss = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weights)
            bce_loss = Reconstruction_loss(recon_x, x.view(-1, self.autoencoder_in_size))

            # compute other loss
            # [B, 300] * [300, size]
            classification_loss = torch.nn.BCELoss(reduction="sum")

            emb_onehot = torch.mm(action.squeeze(0), self.mask)
            emb_domain = emb_onehot[:, :17]
            emb_action = emb_onehot[:, 17: 17+34]
            emb_slot = emb_onehot[:, 34+17:]

            d_loss_1 = classification_loss(d_pred_1, emb_domain)
            a_loss_1 = classification_loss(a_pred_1, emb_action)
            s_loss_1 = classification_loss(s_pred_1, emb_slot)

            return bce_loss, d_loss_1, a_loss_1, s_loss_1

        elif self.config.vae_loss == 'mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).sum()
            raise ValueError("you did not implement this one in this function!!")
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))

    def ae_loss_parallel(self, recon_x, x, d_pred_1, a_pred_1, s_pred_1, d_pred_2, a_pred_2, s_pred_2, action_pred, action):
        """
        Args:
            recon_x: reconstructed bf
            x: original x
            action: action for computing classification loss
            repeated d, a, s
        Returns: loss
        """
        if self.config.vae_loss == 'bce':
            pos_weights = torch.full([392], 15).to(device)
            Reconstruction_loss = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weights)
            bce_loss = Reconstruction_loss(recon_x, x.view(-1, self.autoencoder_in_size))

            # compute other loss
            # [B, 300] * [300, size]
            classification_loss = torch.nn.BCELoss(reduction="sum")

            emb_onehot = torch.mm(action.squeeze(0), self.mask)
            emb_domain = emb_onehot[:, :17]
            emb_action = emb_onehot[:, 17: 17+34]
            emb_slot = emb_onehot[:, 34+17:]

            d_loss_1 = classification_loss(d_pred_1, emb_domain)
            a_loss_1 = classification_loss(a_pred_1, emb_action)
            s_loss_1 = classification_loss(s_pred_1, emb_slot)

            d_loss_2 = classification_loss(d_pred_2, emb_domain)
            a_loss_2 = classification_loss(a_pred_2, emb_action)
            s_loss_2 = classification_loss(s_pred_2, emb_slot)


            action_loss = classification_loss(action_pred, action)
            return bce_loss, d_loss_1, a_loss_1, s_loss_1, d_loss_2, a_loss_2, s_loss_2, action_loss

        elif self.config.vae_loss == 'mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).sum()
            raise ValueError("you did not implement this one in this function!!")
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))

    def eval_hard(self, input, target):
        """
        Args:
            input:
            target:

        Returns: one tensor

        """
        test_loss_hard = tensor([])
        for i in range(input.size(0)):
            left = input[i].unsqueeze(0)
            right = target[i].unsqueeze(-1)
            product = torch.matmul(left, right)
            test_loss_hard = torch.cat((test_loss_hard, product), dim=0)
        return torch.sum(test_loss_hard)

    def get_vae_embed(self, state_action_rep):
        """
        Args:
            state_action_rep: [B, 392]
        Returns: embedding states
        """
        if self.config.vae:
            mean, _ = self.vae.encode(state_action_rep)
            return mean
        else:
            states_emb = self.vae.get_embed(state_action_rep)
            return states_emb

    def _get_label(self):
        """
        Returns: return three dictionary
        """
        # Todo, to do the satistical stuff.
        mask = self.mask.clone()
        pass
    """
    parallel VAE codes over here. group VAE.
    """
    def get_label(self, action):
        emb_onehot = torch.mm(action.squeeze(0), self.mask)
        emb_domain = emb_onehot[:, :17]
        emb_action = emb_onehot[:, 17: 17 + 34]
        emb_slot = emb_onehot[:, 34 + 17:]

        label_d = torch.topk(emb_domain, 1)[1].squeeze(-1)
        label_a = torch.topk(emb_action, 1)[1].squeeze(-1)
        label_s = torch.topk(emb_slot, 1)[1].squeeze(-1)
        return (label_d, label_a, label_s)

    def kl_anneal_function(self, epoch, k, start_epoch, stop_epoch, anneal_function = "linear"):
        """
        Args:
            step: current epoch of this one.
            k:
            x0:
            anneal_function:
        Returns: wight of ELBO loss
        """
        if epoch < start_epoch:
            return 0.
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(epoch-stop_epoch))))

        elif anneal_function == 'linear':
            # when epoch meets the stop_epoch, it will change to the whole KL loss.
            weight = min(1, (epoch - start_epoch)/float(stop_epoch))
            return weight

    def reparameterize(self, training, mu, logvar):
        # TODO, figure how this one will affect this stuff.
        if training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
            pass
        else:
            return mu

    def accumulate_group_evidence(self, class_mu, class_logvar, labels_batch):
        """
        Args:
            class_mu: mean
            class_logvar: variance
            labels_batch: labes in tensor [1,2,3,1,6,1,6,1]
        Returns: content_mu, content_logvar, list_groups_labels (totally different labels), sizes_group (number of each group)
        """
        # convert logvar to variance for calculations
        content_mu = []
        content_inv_logvar = []
        list_groups_labels = []
        sizes_group = []
        groups = (labels_batch).unique()
        # calculate var inverse for each group using group vars
        for _, g in enumerate(groups):
            group_label = g.item()
            samples_group = labels_batch.eq(group_label).nonzero().squeeze()

            if samples_group.numel() > 0:
                inv_group_logvar = - class_logvar[samples_group, :]
                # multiply by inverse variance
                inv_group_var = torch.exp(inv_group_logvar)
                group_mu = class_mu[samples_group, :] * inv_group_var

                if samples_group.numel() > 1:
                    group_mu = group_mu.sum(0, keepdim=True)
                    inv_group_logvar = torch.logsumexp(inv_group_logvar,
                                                       dim=0, keepdim=True)
                else:
                    group_mu = group_mu[None, :]
                    inv_group_logvar = inv_group_logvar[None, :]

                content_mu.append(group_mu)
                content_inv_logvar.append(inv_group_logvar)
                list_groups_labels.append(group_label)
                sizes_group.append(samples_group.numel())

        content_mu = torch.cat(content_mu, dim=0)
        content_inv_logvar = torch.cat(content_inv_logvar, dim=0)
        sizes_group = torch.FloatTensor(sizes_group)
        # inverse log variance
        content_logvar = - content_inv_logvar
        # multiply group var with group log variance
        content_mu = content_mu * torch.exp(content_logvar)
        return content_mu, content_logvar, list_groups_labels, sizes_group

    # def group_wise_reparameterize(self, training, mu, logvar, labels_batch):
    #     eps_dict = {}
    #     # generate only 1 eps value per group label
    #     for label in torch.unique(labels_batch):
    #         eps_dict[label.item()] = torch.cuda.FloatTensor(1, logvar.size(1)).normal_(0., 0.1).to(device)
    #
    #     if training:
    #         std = logvar.mul(0.5).exp_()
    #         reparameterized_var = Variable(std.data.new(std.size()))
    #
    #         # multiply std by correct eps and add mu
    #         for i in range(logvar.size(0)):
    #             reparameterized_var[i] = std[i].mul(Variable(eps_dict[labels_batch[i].item()]))
    #             reparameterized_var[i].add_(mu[i])
    #
    #         return reparameterized_var
    #     else:
    #         return mu
    def group_wise_reparameterize_each(self, training, mu, logvar, labels_batch, list_groups_labels, sizes_group):
        eps_dict = {}
        batch_size = labels_batch.size(0)

        if training:
            std = logvar.mul(0.5).exp_()
        else:
            std = torch.zeros_like(logvar)

        content_samples = []
        indexes = []
        sizes = []
        # multiply std by correct eps and add mu
        for i, g in enumerate(list_groups_labels):
            samples_group = labels_batch.eq(g).nonzero().squeeze()
            size_group = samples_group.numel()
            assert size_group == sizes_group[i]
            if size_group > 0:
                eps = torch.cuda.FloatTensor(size_group, std.size(1)).normal_()

                group_content_sample = std[i][None, :] * eps + mu[i][None, :]
                content_samples.append(group_content_sample)
                if size_group == 1:
                    samples_group = samples_group[None]
                indexes.append(samples_group)
                size_group = torch.ones(size_group) * size_group
                sizes.append(size_group)

        content_samples = torch.cat(content_samples, dim=0)
        indexes = torch.cat(indexes)
        sizes = torch.cat(sizes)

        return content_samples, indexes, sizes

    def vae_3_loss(self, recon_x, x):
        """
        Args:
            recon_x:
            x:
        Returns:
        """
        pos_weights = torch.full([392], 15).to(device)
        Reconstruction_loss = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weights)
        BCE_loss = Reconstruction_loss(recon_x, x.view(-1, self.autoencoder_in_size))

        return BCE_loss


class WGanAgent_VAE_State(GanAgent_AutoEncoder_State):
    # only state is fed to VAE, action is onehot with 300 dims
    def __init__(self, corpus, config, action2name):
        super(WGanAgent_VAE_State, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_sat.WoZDiscriminator(config)
        self.generator =gan_model_sat.WoZGenerator_StateVae(config)
        self.autoencoder_in_size = config.state_out_size 
        self.vae = gan_model_vae.VAE(config)
        
    def gen_train(self, sample_shape):
        state_rep, action_rep =self.sample_step(sample_shape)
        disc_v = self.discriminator.forward_wgan(state_rep, action_rep)
        gen_loss = -torch.mean(disc_v)
        gen_loss = Pack(gen_loss= gen_loss)
        return gen_loss, (state_rep.detach(), action_rep.detach())

    def gen_validate(self, sample_shape, record_gen=False):
        state_rep, action_rep =self.sample_step(sample_shape)
        # policy_prob = self.policy_validate(state_rep, action_rep)
        if record_gen:
            return state_rep, action_rep
        else:
            return [], []

    def disc_train(self, sample_shape, batch_feed, fake_state_action=None):
        batch_size = sample_shape[0]
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        embed_batch = self.vae.get_embed(self.cast_gpu(state_rep))
        real_disc_v = self.discriminator.forward_wgan(embed_batch.detach(), self.cast_gpu(action_rep))

        if fake_state_action is None:
            fake_state_rep, fake_action_rep = self.sample_step(sample_shape)
        else:
            fake_state_rep, fake_action_rep = fake_state_action
        fake_disc_v = self.discriminator.forward_wgan(fake_state_rep.detach(), fake_action_rep.detach())

        real_disc_loss = - torch.mean(real_disc_v) 
        fake_disc_loss = torch.mean(fake_disc_v)
        disc_loss = real_disc_loss + fake_disc_loss
        disc_loss = Pack(disc_loss=disc_loss)
        disc_acc = np.array([-real_disc_loss.item(), fake_disc_loss.item()])
        return disc_loss, disc_acc


    def vae_train(self, batch_feed):
        # state_rep, action_rep = self.read_real_data_onehot(None, batch_feed)
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        state_rep = self.cast_gpu(state_rep)
        recon_batch, mu, logvar = self.vae(state_rep)
        loss = self.vae_loss(recon_batch, state_rep, mu, logvar)
        vae_loss = Pack(vae_loss=loss) 
        return vae_loss

    
    def vae_loss(self, recon_x, x, mu, logvar):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size), reduction='sum')
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).sum()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l2_loss = self.vae.l2_norm()
        loss = (BCE + KLD)/len(x.view(-1))
        return loss + l2_loss

    def get_vae_embed(self, state_action_rep):
        mean, _ = self.vae.encode(state_action_rep)
        return mean

##################################################################
########### In this agent, the state and action are fed to the VAE together.
class GanAgent_VAE_StateActioneEmbed(GanAgent_AutoEncoder):
    def __init__(self, corpus, config, action2name):
        super(GanAgent_VAE_StateActioneEmbed, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_sat.WoZDiscriminator_StateActionEmbed(config)
        self.generator =gan_model_sat.WoZGenerator_StateActionEmbed(config)
        self.autoencoder_in_size = config.state_out_size + 100
        # self.vae = gan_model_vae.VAE_StateActionEmbed(config) # the input size is 392 + 100
        self.vae = gan_model_vae.VAE_StateActionEmbedMerged(config) # the input size is 392 + 100



    def read_real_data_onehot_300(self, sample_shape, batch_feed):
        action_data_feed = one_hot_embedding(batch_feed['action_id'], 300)
        action_rep_seg = self.np2var(batch_feed['action_rep_seg'], FLOAT).view(-1, 100)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_rep_seg

    def vae_train(self, batch_feed):
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        state_action_rep = torch.cat([state_rep, action_rep],-1)
        recon_batch, mu, logvar = self.vae(state_action_rep)
        loss = self.vae_loss(recon_batch, state_action_rep, mu, logvar)
        vae_loss = Pack(vae_loss=loss) 
        return vae_loss

    
    def vae_loss(self, recon_x, x, mu, logvar):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size), reduction='sum')
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).sum()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l2_loss = self.vae.l2_norm()
        loss = (BCE + KLD)/len(x.view(-1))
        return loss + l2_loss

    def get_vae_embed(self, state_action_rep):
        mean, _ = self.vae.encode(state_action_rep)
        return mean

    
    def autoencoder_loss(self, recon_x, x):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size))
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).mean()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        l2_loss = self.vae.l2_norm()
        loss = BCE + l2_loss
        return loss + l2_loss
    
    
    def sample_step(self, sample_shape):
        batch_size, state_noise_dim, action_noise_dim = sample_shape
        z_noise = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, state_noise_dim)))))
        state_action_rep= self.generator(z_noise)
        return state_action_rep

    
    def gan_vae_optimizer(self, config):
        params = list(self.generator.parameters()) + list(self.vae.parameters())
        if config.op == 'adam':
            print("Use adam")
            return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           params), lr=config.init_lr, betas=(0.5, 0.999))
        elif config.op == 'sgd':
            print("Use SGD")
            return torch.optim.SGD(params, lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            print("RMSProp")
            return torch.optim.RMSprop(params, lr=config.init_lr,
                                       momentum=config.momentum)
        
    def gen_train(self, sample_shape):
        state_action_rep =self.sample_step(sample_shape)
        mean_dist_1 = - (state_action_rep.mean(dim=0) - state_action_rep).pow(2).mean() * self.config.sim_factor
        mean_dist = mean_dist_1 
        disc_v = self.discriminator(state_action_rep)
        gen_loss = -torch.mean(torch.log(disc_v))
        gen_loss = Pack(gen_loss= gen_loss, mean_dist = mean_dist)
        return gen_loss, state_action_rep.detach()
    
        
    def disc_train(self, sample_shape, batch_feed, fake_state_action=None):
        batch_size = sample_shape[0]
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        state_action = torch.cat([self.cast_gpu(state_rep), self.cast_gpu(action_rep)], -1)
        embed_batch = self.vae.get_embed(self.cast_gpu(state_action))
        real_disc_v = self.discriminator(embed_batch.detach())
        if fake_state_action is None:
            fake_state_action = self.sample_step(sample_shape)
        else:
            fake_state_action = fake_state_action
        state_rep_f = fake_state_action
        fake_disc_v = self.discriminator(state_rep_f.detach())
        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((batch_size,),1.0), torch.full((batch_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        # disc_loss = BCELoss_double(rf_disc_v.view(-1), labels)
        # disc_loss = - torch.mean(real_disc_v) + torch.mean(fake_disc_v)
        disc_loss = Pack(disc_loss=disc_loss)
        disc_acc = cal_accuracy(rf_disc_v.view(-1), labels_one)
        return disc_loss, disc_acc
    
    def gen_validate(self, sample_shape, record_gen=False):
        state_action_rep =self.sample_step(sample_shape)
        # policy_prob = self.policy_validate(state_rep, action_rep)
        if record_gen:
            return [], state_action_rep
        else:
            return [], []

###########################################################
###########   
###########################################################
"""
This is the best model I have seen.
"""
class GanAgent_StateVaeActionSeg(GanAgent_AutoEncoder_State):
    # only state is fed to VAE, the action is the concatenation with (domain, act, slot), rather than onehot-300
    def __init__(self, corpus, config, action2name):
        super(GanAgent_StateVaeActionSeg, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_sat.WoZDiscriminator_StateVaeActionSeg(config)
        self.generator =gan_model_sat.WoZGenerator_StateVaeActionSeg(config)
        self.autoencoder_in_size = config.state_out_size 
        self.vae = gan_model_vae.VAE(config)

    def read_real_data_onehot_300(self, sample_shape, batch_feed):
        # the action rep should be the concatenated version, which has dimension 160
        action_data_feed = self.np2var(batch_feed['action_rep_seg'], FLOAT).view(-1, 160)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed

    def vae_train(self, batch_feed):
        # state_rep, action_rep = self.read_real_data_onehot(None, batch_feed)
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        state_rep = self.cast_gpu(state_rep)
        recon_batch, mu, logvar = self.vae(state_rep)
        loss = self.vae_loss(recon_batch, state_rep, mu, logvar)
        vae_loss = Pack(vae_loss=loss) 
        return vae_loss
    

    
    def vae_loss(self, recon_x, x, mu, logvar):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size), reduction='sum')
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).sum()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l2_loss = self.vae.l2_norm()
        loss = (BCE + KLD)/len(x.view(-1))
        return loss + l2_loss

    def get_vae_embed(self, state_action_rep):
        mean, _ = self.vae.encode(state_action_rep)
        return mean

    def disc_train(self, sample_shape, batch_feed, fake_state_action=None):
        batch_size = sample_shape[0]
        # state_rep, action_rep = self.read_real_data(None, batch_feed)
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        # state_action = torch.cat([self.cast_gpu(state_rep), self.cast_gpu(action_rep)], -1)
        embed_batch = self.vae.get_embed(self.cast_gpu(state_rep))
        real_disc_v = self.discriminator(embed_batch.detach(), self.cast_gpu(action_rep))
        if fake_state_action is None:
            fake_state_action = self.sample_step(sample_shape)
        else:
            fake_state_action = fake_state_action
        state_rep_f, action_rep_f = fake_state_action
        if np.random.random()<0.5:
            # print(action_rep.size())
            # state_rep_f, action_rep_f = embed_batch.detach(), self.shuffle_action(action_rep)
            state_rep_f, action_rep_f = embed_batch.detach(), action_rep[torch.randperm(action_rep.size()[0]), :] 
            # print(action_rep_f.size())
        fake_disc_v = self.discriminator(state_rep_f.detach(), action_rep_f.detach())
        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((batch_size,),1.0), torch.full((batch_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        # disc_loss = BCELoss_double(rf_disc_v.view(-1), labels)
        # disc_loss = - torch.mean(real_disc_v) + torch.mean(fake_disc_v)
        disc_loss = Pack(disc_loss=disc_loss)
        disc_acc = cal_accuracy(rf_disc_v.view(-1), labels_one)
        return disc_loss, disc_acc
    
