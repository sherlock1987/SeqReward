# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
# from laed.models.model_bases import summary
from torch.autograd import Variable
import torch
# from laed.dataset.corpora import PAD, EOS, EOT
# from laed.enc2dec.decoders import TEACH_FORCE, GEN, DecoderRNN
# from laed.utils import get_dekenize
import os
from collections import defaultdict
import logging
# from torch_utils import one_hot_embedding
from utils import print_accuracy
logger = logging.getLogger()

"""
This code is for the validation for G and D, also for VAE stuff.

"""

class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, loss):
        for key, val in loss.items():
            # logger.info("sssss",val)
            if val is not None and type(val) is not bool:
                
                self.losses[key].append(val.item())

    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.item())

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def pprint(self, name, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
            str_losses.append("{} {:.3f}".format(key, avg_loss))
            if 'nll' in key:
                str_losses.append("PPL {:.3f}".format(np.exp(avg_loss)))
        if prefix:
            return "{}: {} {}".format(prefix, name, " ".join(str_losses))
        else:
            return "{} {}".format(name, " ".join(str_losses))

    def avg_loss(self):
        return np.mean(self.backward_losses)

    def get_number(self, key):
        """
        Args:
            key:

        Returns: list

        """
        return self.losses[key]

    def get_average(self, key):
        """
        Args:
            key: current key.
        Returns: one number average
        """
        content = self.losses[key]
        return np.mean(content)

"""
1. use all of the validation data to validate, even if it should be all 1.
"""
def disc_validate(agent, valid_feed, config, sample_shape, D_criterions, batch_cnt=None , training_mode = "000"):
    """
    Args:
        agent:
        valid_feed:
        config:
        sample_shape:
        batch_cnt:
        D_criterions:
        training_mode:
    Returns:
    """
    with torch.no_grad():
        agent.eval()
        valid_feed.epoch_init(config, shuffle=False, verbose=False)

        losses = LossManager()
        acc_feed = np.array([0,0,0,0,0.0,0.0])
        batch_num = 0
        while True:
            batch = valid_feed.next_batch()
            if batch is None:  
                break                                                                                                                     
            loss, acc = agent.disc_train(sample_shape, batch, D_criterions= D_criterions, training_mode = training_mode)
            # wgan_reward.append(torch.stack(acc))
            acc_feed = acc_feed + acc
            losses.add_loss(loss)
            losses.add_backward_loss(agent.discriminator.model_sel_loss(loss, batch_cnt))
            batch_num+=1
    valid_loss = losses.avg_loss()
    logger.info(losses.pprint(valid_feed.name))
    logger.info("Total valid loss {}".format(valid_loss))

    if config.gan_type=='gan':
        print_accuracy(acc_feed, batch_num, config)
    else:
        logger.info("Wgan Disc Real and Fake Score: {}, {}".format(acc_feed[0]/batch_num, acc_feed[1]/batch_num))
    return valid_loss

def vae_validate(agent, valid_feed, config, batch_cnt=None):
    with torch.no_grad():
        agent.vae.eval()
        valid_feed.epoch_init(config, shuffle=False, verbose=False)
        losses = LossManager()
        batch_num = 0
        while True:
            batch = valid_feed.next_batch()
            if batch is None:  
                break                                                                                                                     
            loss= agent.vae_validation(batch)
            # wgan_reward.append(torch.stack(acc))
            losses.add_loss(loss)
            losses.add_backward_loss(agent.vae.model_sel_loss(loss, batch_cnt))            
            batch_num += 1

    vae_eval = (losses.get_average("vae_loss"))/config.batch_size
    d_eval = (losses.get_average("d_loss"))/config.batch_size
    a_eval = (losses.get_average("a_loss"))/config.batch_size
    s_eval = (losses.get_average("s_loss"))/config.batch_size
    action_eval = (losses.get_average("action_loss"))/config.batch_size

    kl_loss_whole = (losses.get_average("ELBO_whole"))/config.batch_size

    # valid_loss = losses.avg_loss()
    # valid_loss /= config.batch_size
    # print all of the name and loss
    logger.info(losses.pprint(valid_feed.name))
    logger.info("Evaluation VAE: {:.3f}(15.00), [d, a, s, action] = [{:.3f}, {:.3f}, {:.3f}, {:.3f}], ELBO: {:.3f}".format(vae_eval, d_eval, a_eval, s_eval, action_eval, kl_loss_whole))
    agent.vae.train()
    # Todo, only use the Reconstruction loss as my main loss function, am I right? sure.
    # valid_loss = vae_eval + d_eval + a_eval + s_eval + action_eval
    return vae_eval,  kl_loss_whole

def disc_validate_for_tsne(agent, machine_rep, valid_feed, config, sample_shape):
    # this function may be removed later
    with torch.no_grad():
        agent.eval()
        pred_machine = []
        pred_human = []
        disc_value = np.array([0.0,0.0])
        valid_feed.epoch_init(config, shuffle=False, verbose=True)
        for batch_id, batch in enumerate(machine_rep):
            state_rep, action = batch
            disc_v = agent.discriminator(state_rep, action)
            label = [1 if v>0.5 else 0 for v in disc_v.view(-1).tolist()]
            pred_machine = pred_machine + label
            disc_value[1] += disc_v.mean().item()  
        while True:
            batch = valid_feed.next_batch()
            if batch is None:  
                break 
            real_state_rep, action_data_feed = agent.read_real_data(sample_shape, batch)
            real_disc_v = agent.discriminator(real_state_rep, action_data_feed)
            label = [1 if v>0.5 else 0 for v in disc_v.view(-1).tolist()]
            pred_human = pred_human + label 
            disc_value[0] += real_disc_v.mean().item()
        disc_value = disc_value/len(machine_rep)
        logger.info("Average disc value for human and machine: {:.3f}, {:.3f}".format(disc_value[0], disc_value[1]))
        return [pred_human, pred_machine], disc_value         

def disc_validate_for_tsne_single_input(agent, machine_rep, valid_feed, config, sample_shape):
    """
    Args:
        agent:
        machine_rep: random_action with fixed states
        valid_feed: fixed states
        config:
        sample_shape:
    Returns:[pred_human, pred_machine], disc_value
    """
    # this function may be removed later
    with torch.no_grad():
        agent.eval()
        pred_machine = []
        pred_human = []
        disc_value = np.array([0.0,0.0])
        valid_feed.epoch_init(config, shuffle=False, verbose=False)
        """
        machine
        """
        for batch_id, batch in enumerate(machine_rep):
            state_rep, action_rep = batch
            # state_action = torch.cat([agent.cast_gpu(state_rep), agent.cast_gpu(action_rep)], -1)
            state_action = agent.cast_gpu(state_rep)
            state_rep = agent.get_vae_embed(state_action)
            if config.gan_type=='gan':
                disc_v = agent.discriminator(state_rep, action_rep)
                disc_v = torch.mean((disc_v), dim = -1)
            else:
                disc_v = agent.discriminator.forward_wgan(state_rep, action_rep)
            label = [1 if v>0.5 else 0 for v in disc_v.view(-1).tolist()]
            pred_machine = pred_machine + label
            disc_value[1] += disc_v.mean().item()
        """
        human
        """
        while True:
            batch = valid_feed.next_batch()
            if batch is None:  
                break 
            # state_rep, action_rep = agent.read_real_data(sample_shape, batch)
            state_rep, action_rep = agent.read_real_data_onehot_300(sample_shape, batch)
            state_action = agent.cast_gpu(state_rep)
            # state_action = torch.cat([agent.cast_gpu(state_rep), agent.cast_gpu(action_rep)], -1)
            state_action_rep = agent.get_vae_embed(state_action)
            if config.gan_type=='gan':
                real_disc_v = agent.discriminator(state_action_rep, action_rep)
                real_disc_v = torch.mean((real_disc_v), dim = -1)
            else:
                real_disc_v = agent.discriminator.forward_wgan(state_action_rep, action_rep)
            label = [1 if v>0.5 else 0 for v in real_disc_v.view(-1).tolist()]
            pred_human = pred_human + label 
            disc_value[0] += real_disc_v.mean().item()
        disc_value = disc_value/len(machine_rep)
        logger.info("Average disc value for human and machine: {:.3f}, {:.3f}".format(disc_value[0], disc_value[1]))
        return [pred_human, pred_machine], disc_value                                                                                       

def disc_validate_for_tsne_state_action_embed(agent, machine_rep, valid_feed, config, sample_shape):
    # this function may be removed later
    with torch.no_grad():
        agent.eval()
        pred_machine = []
        pred_human = []
        disc_value = np.array([0.0,0.0])
        valid_feed.epoch_init(config, shuffle=False, verbose=True)
        for batch_id, batch in enumerate(machine_rep):
            state_rep, action_rep = batch
            state_action = torch.cat([agent.cast_gpu(state_rep), agent.cast_gpu(action_rep)], -1)
            state_rep = agent.get_vae_embed(state_action)
            disc_v = agent.discriminator(state_rep)
            label = [1 if v>0.5 else 0 for v in disc_v.view(-1).tolist()]
            pred_machine = pred_machine + label
            disc_value[1] += disc_v.mean().item()  
        while True:
            batch = valid_feed.next_batch()
            if batch is None:  
                break 
            # state_rep, action_rep = agent.read_real_data(sample_shape, batch)
            state_rep, action_rep = agent.read_real_data_onehot_300(sample_shape, batch)
            state_action = torch.cat([agent.cast_gpu(state_rep), agent.cast_gpu(action_rep)], -1)
            state_action_rep = agent.get_vae_embed(state_action)
            real_disc_v = agent.discriminator(state_action_rep)
            label = [1 if v>0.5 else 0 for v in disc_v.view(-1).tolist()]
            pred_human = pred_human + label 
            disc_value[0] += real_disc_v.mean().item()
        disc_value = disc_value/len(machine_rep)
        logger.info("Average disc value for human and machine: {:.3f}, {:.3f}".format(disc_value[0], disc_value[1]))
        return [pred_human, pred_machine], disc_value                                                                                       

def policy_validate_for_human(agent, valid_feed, config, sample_shape, batch_cnt=None):
    with torch.no_grad():
        agent.eval()
        valid_feed.epoch_init(config, shuffle=False, verbose=True)
        policy_prob = []
        while True:
            batch = valid_feed.next_batch()
            if batch is None:  
                break                                                                                                                     
            prob_batch = agent.policy_validate_for_human(sample_shape, batch)
            # wgan_reward.append(torch.stack(acc))
            policy_prob.append(prob_batch)
    print(torch.stack(policy_prob).mean())
"""
1. sample some data and use D to give log score.
"""
def gen_validate(agent, valid_feed, config, sample_shape, D_criterions, done_epoch=0, batch_cnt=None, training_mode = "000"):
    """
    Args:
        agent:
        valid_feed: no use, just use its name.
        config:
        sample_shape:
        D_criterions:
        done_epoch:
        batch_cnt:
    Returns: valid_loss, record_gen_data
    """
    with torch.no_grad():
        agent.eval()
        valid_feed.epoch_init(config, shuffle=False, verbose=False)
        losses = LossManager()
        # randomly sample 100 batches 
        sample_count = 0
        policy_prob = 0
        record_gen_data = []
        record_flag=False

        if done_epoch % config.record_step == 0 or done_epoch==-1:
            record_flag=True

        # use all of the data to do validation.
        # 7365 //

        while sample_count < (7300//config.batch_size):
            batch = valid_feed.next_batch()
            if batch is None: break
            loss, _ = agent.gen_train(batch, sample_shape, training_mode = training_mode)
            _, sampled_batch = agent.gen_validate(sample_shape, record_flag)
            record_gen_data.append(sampled_batch)
            # policy_prob += prob_batch
            losses.add_loss(loss)
            losses.add_backward_loss(agent.generator.model_sel_loss(loss, batch_cnt))
            sample_count += 1

    valid_loss = losses.avg_loss()
    logger.info(losses.pprint(valid_feed.name))
    logger.info("Total valid loss {}".format(valid_loss))
    # logger.info("Total log_y from previous policy {}".format(policy_prob/sample_count))
    if record_flag:
        return valid_loss, record_gen_data
    else:
        return valid_loss, []

def build_fake_data(agent, valid_feed, config, sample_shape, batch_cnt=None):
    """
    This function will return machine: human = [state, action_randperm] : [state, action_noshuffle]
    Args:
        agent:
        valid_feed:
        config:
        sample_shape:
        batch_cnt:
    Returns:
    """
    with torch.no_grad():
        agent.eval()
        valid_feed.epoch_init(config, shuffle=False, verbose=True)
        machine_rep = []
        human_rep = []
        while True:
            batch = valid_feed.next_batch()
            if batch is None:  
                break  
            if config.domain=='movie' or config.input_type=='sat':
                real_state_rep, action_data_feed = agent.read_real_data(sample_shape, batch)

            elif config.domain=='multiwoz':
                real_state_rep, action_data_feed = agent.read_real_data_onehot_300(sample_shape, batch)
            else:
                raise ValueError("no such domain: {}".format(config.domain))
            machine_state_rep =real_state_rep   
            # machine_state_rep =real_state_rep[:,torch.randperm(real_state_rep.size()[1])] 
            # machine_action_rep =action_data_feed[:,torch.randperm(action_data_feed.size()[1])]   
            machine_action_rep = action_data_feed[torch.randperm(action_data_feed.size()[0]), :]

                
            # machine_action_rep = shuffle_action(action_data_feed)
            machine_rep.append((machine_state_rep, machine_action_rep))
            human_rep.append((real_state_rep.cpu(), action_data_feed.cpu()))
        return machine_rep, human_rep

def disc_train_history(agent, batch, sample_shape, experience_pool, disc_optimizer, times=2):
    for _ in range(times):
        fake_s_a = experience_pool.next()
        disc_optimizer.zero_grad()
        disc_loss, train_acc = agent.disc_train(sample_shape, batch, fake_s_a)
        agent.discriminator.backward(0, disc_loss)
        disc_optimizer.step()

def shuffle_action(x):
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

