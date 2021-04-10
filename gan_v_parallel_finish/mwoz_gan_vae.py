# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import argparse
import logging
import os
import json
import torch
import random
import numpy as np
from gan import gan_main as engine
from laed.models.model_bases import summary

from gan.gan_agent import GanAgent_VAE_State, WGanAgent_VAE_State
from gan.torch_utils import  weights_init
from gan.utils import load_context_encoder, load_action2name, save_data_for_tsne, load_model_vae
from laed.utils import str2bool, prepare_dirs_loggers, get_time, process_config, zip_folder
from laed.dataset.data_loaders import WoZGanDataLoaders
from gan.gan_validate import disc_validate, gen_validate, policy_validate_for_human, build_fake_data
from collections import defaultdict

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mask_onehot_path = os.path.join(root_dir, "gan_v/mask_python2_onehot.pkl")
mask_path = os.path.join(root_dir, "gan_v/mask.pkl")
"""
1. Config_file
2. main 
        Train VAE
        Train GAN
"""

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

"""
Config file is over here.
"""
# Method
method_arg = add_argument_group('Method')
method_arg.add_argument('--gan_type', type=str, default='gan', help="gan_type: gan, wgan")
method_arg.add_argument('--state_type', type=str, default='table', help="state_type: rnn, table")
method_arg.add_argument('--input_type', type=str, default='sa', help="sa: state_action, sat: gumble representation")
method_arg.add_argument('--bucket_num', type=int, default=4)  # if input_type is 'sat', how many categories will a table state have

method_arg.add_argument('--vae', type=bool, default=False)
method_arg.add_argument('--vae_max_patience', type = int, default = 100)
# True will enable vae training during adversarial training
method_arg.add_argument('--vae_encoder', type=str, default="gru", help = "[gru, rnn]")
method_arg.add_argument('--vae_train', type=bool, default=True)
method_arg.add_argument('--vae_loss', type=str, default='bce', help="bce, mean_sq")
method_arg.add_argument('--round_for_disc', type=str2bool, default=False)


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_dir', type=str, nargs='+', default=['data/artificial'])
data_arg.add_argument('--log_dir', type=str, default='logs')
data_arg.add_argument('--domain', type=str, default='multiwoz')

data_arg.add_argument('--action_id_path', type=str, nargs='+', default='2019-06-26T07-08-art_vst_laed.py/2019-06-26T07-08-z.pkl.cluster_id.json')
data_arg.add_argument('--action2name_path', type=str, nargs='+', default='2019-06-26T07-08-art_vst_laed.py/2019-06-26T07-08-z.pkl.id_cluster.json')
data_arg.add_argument('--load_sess', type=str, default="2018-02-04T01-20-45")
data_arg.add_argument('--encoder_sess', type=str, default="2019-06-26T07-08-art_vst_laed.py")
# for mask stuff
data_arg.add_argument("--mask_onehot_path", type = str, default = mask_onehot_path)
data_arg.add_argument("--mask_path", type = str, default = mask_path)

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--hier', type=str2bool, default=True)
net_arg.add_argument('--y_size', type=int, default=4)  # number of discrete variables
net_arg.add_argument('--k', type=int, default=3)  # number of classes for each variable
net_arg.add_argument('--action_num', type=int, default=300)  
net_arg.add_argument('--gan_ratio', type=int, default=1)
net_arg.add_argument('--gan_ratio_g', type=int, default=1)
net_arg.add_argument('--state_noise_dim', type=int, default=128)  
net_arg.add_argument('--action_noise_dim', type=int, default=128)  
net_arg.add_argument('--state_in_size', type=int, default=120)  
# net_arg.add_argument('--state_out_size', type=int, default=680)  
net_arg.add_argument('--state_out_size', type=int, default=392)  
net_arg.add_argument('--vae_embed_size', type=int, default=64)
# , helper = "to control the variance of generated states and action")
net_arg.add_argument('--sim_factor', type=float, default=0.0)
net_arg.add_argument('--use_attribute', type=str2bool, default=True)
net_arg.add_argument('--rnn_cell', type=str, default='gru')
net_arg.add_argument('--embed_size', type=int, default=20)

net_arg.add_argument('--utt_type', type=str, default='attn_rnn')
net_arg.add_argument('--utt_cell_size', type=int, default=100)
net_arg.add_argument('--ctx_cell_size', type=int, default=100)
net_arg.add_argument('--dec_cell_size', type=int, default=100)

net_arg.add_argument('--enc_cell_size', type=int, default=100)

net_arg.add_argument('--bi_ctx_cell', type=str2bool, default=False)
net_arg.add_argument('--bi_enc_cell', type=str2bool, default=False)
# net_arg.add_argument('--max_utt_len', type=int, default=100)
net_arg.add_argument('--max_utt_len', type=int, default=40)
net_arg.add_argument('--max_dec_len', type=int, default=20)
net_arg.add_argument('--max_vocab_cnt', type=int, default=1000)
net_arg.add_argument('--num_layer', type=int, default=1)

net_arg.add_argument('--use_attn', type=str2bool, default=False)
net_arg.add_argument('--attn_type', type=str, default='cat')
net_arg.add_argument('--use_mutual', type=str2bool, default=True)
net_arg.add_argument('--use_reg_kl', type=str2bool, default=True)
net_arg.add_argument('--greedy_q', type=str2bool, default=True)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--wgan_w_clip', type=float, default = 0.01)
train_arg.add_argument('--op', type=str, default='adam')
# train_arg.add_argument('--op', type=str, default='rmsprop')
train_arg.add_argument('--backward_size', type=int, default = 40)
train_arg.add_argument('--step_size', type=int, default=1)
# train_arg.add_argument('--grad_clip', type=float, default=3.0)
train_arg.add_argument('--init_w', type=float, default=0.08)
train_arg.add_argument('--init_lr', type=float, default=0.0001)
train_arg.add_argument('--momentum', type=float, default=0.0)
train_arg.add_argument('--lr_hold', type=int, default=1)
train_arg.add_argument('--lr_decay', type=float, default=0.6)
train_arg.add_argument('--l2_lambda', type=float, default=0.0001)
train_arg.add_argument('--dropout', type=float, default=0.3)
# This is to avoid gradient boosting.
train_arg.add_argument('--clip', type=float, default = 7.0)
train_arg.add_argument('--improve_threshold', type=float, default = 0.996)
train_arg.add_argument('--patient_increase', type=float, default = 2.0)


train_arg.add_argument('--early_stop', type=str2bool, default = True)
train_arg.add_argument('--max_epoch', type=int, default = 3000)
train_arg.add_argument('--max_epoch_list', type=list, default = [3000, 300, 300, 300, 300, 300, 300])
train_arg.add_argument("--max_patience", type = int, default = 30)
train_arg.add_argument('--max_vocab_size', type=int, default = 1000)
train_arg.add_argument('--gumbel_temp', type=float, default = 0.8, help = "temperate")

# MISC
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--save_model', type=str2bool, default=True)
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--print_step', type=int, default=500)
misc_arg.add_argument('--fix_batch', type=str2bool, default=False)
misc_arg.add_argument('--train_prior', type=str2bool, default=False)
misc_arg.add_argument('--ckpt_step', type=int, default=1000)
misc_arg.add_argument('--record_step', type=int, default=16)
misc_arg.add_argument('--batch_size', type=int, default=64)
misc_arg.add_argument('--preview_batch_num', type=int, default=1)
misc_arg.add_argument('--gen_type', type=str, default='greedy')
misc_arg.add_argument('--avg_type', type=str, default='word')
misc_arg.add_argument('--beam_size', type=int, default=10)
misc_arg.add_argument('--seed', type=int, default=99)
misc_arg.add_argument('--forward_only', type=str2bool, default=False)


logger = logging.getLogger()


def main(config):
    prepare_dirs_loggers(config, os.path.basename(__file__))
    manualSeed=config.seed
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    sample_shape = config.batch_size, config.state_noise_dim, config.action_noise_dim

    # evaluator = evaluators.BleuEvaluator(os.path.basename(__file__))
    evaluator = False

    train_feed = WoZGanDataLoaders("train", config)
    valid_feed = WoZGanDataLoaders("val", config)
    test_feed = WoZGanDataLoaders("test", config)


 # action2name = load_action2name(config)
    action2name = None
    corpus_client = None

    # Todo, finisht the part of Wgan
    if config.gan_type=='wgan':
        model = WGanAgent_VAE_State(corpus_client, config, action2name)
    else:
        model = GanAgent_VAE_State(corpus_client, config, action2name)
    
    logger.info(summary(model, show_weights=False))
    model.discriminator.apply(weights_init)
    model.generator.apply(weights_init)
    model.vae.apply(weights_init)

    if config.forward_only:
        test_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.log_dir, config.load_sess, "model")
    else:
        test_file = os.path.join(config.session_dir,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file = os.path.join(config.session_dir, "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.session_dir, "model")
        vocab_file = os.path.join(config.session_dir, "vocab.json")

    if config.use_gpu:
        model.cuda()

    pred_list = []
    generator_samples = []
    """
    print("Evaluate initial model on Validate set")
    model.eval()
    # policy_validate_for_human(model,valid_feed, config, sample_shape)
    disc_validate(model, valid_feed, config, sample_shape)
    _, sample_batch = gen_validate(model,valid_feed, config, sample_shape, -1)
    generator_samples.append([-1, sample_batch])
    machine_data, human_data = build_fake_data(model, valid_feed, config, sample_shape)    
    """
    model.train()

    # this is for the training of VAE. If you already have a pretrained model, you can skip this step.
    # print("Start VAE training")
    # if config.forward_only is False:
    #     try:
    #         engine.vae_train(model, train_feed, valid_feed, test_feed, config)
    #     except KeyboardInterrupt:
    #         print("Stopped by myself")
    # print("AutoEncoder Training Finish")
    # load_model_vae(model, config)

    # this is a pretrained vae model, you can load it to the current model.
    # path='./logs/vae_3'
    # gan_v_parallel/logs/vae_3_2_with_KL_mean
    # path = "./logs/vae_3_3_little_KL"
    # path = "./logs/vae_3_3_no_kl"
    path = "./logs/vae_3" # this is the AE model I train in the first place.
    load_model_vae(model, path)
    print("Start GAN training")
    machine_data, human_data = build_fake_data(model, valid_feed, config, sample_shape)

    if config.forward_only is False:
        try:
            engine.gan_train(model, machine_data, train_feed, valid_feed, test_feed, config, evaluator, pred_list, generator_samples)
        except KeyboardInterrupt:
            print("Training stopped by keyboard.")
    print("Reward Model Training Done ! ")
    print("Saved path: {}".format(model_file))

# get config over here
config, unparsed = get_config()
config = process_config(config)

if __name__ == "__main__":
    main(config)
    # zip this folder
    zip_path = config.session_dir + ".zip"
    zip_folder(config.session_dir, zip_path)


