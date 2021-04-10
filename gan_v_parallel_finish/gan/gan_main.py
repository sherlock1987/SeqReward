# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import os
from collections import defaultdict
import logging
from torch_utils import  HistoryData
from utils import  save_model, save_model_woz, save_model_vae
from gan_validate import disc_validate, vae_validate, gen_validate, LossManager, disc_validate_for_tsne_single_input, disc_validate_for_tsne_state_action_embed
logger = logging.getLogger()
import json
def save_best_model(agent):
    pass

def train_disc_with_history(agent, history_pool, batch, sample_shape, dis_optimizer, batch_cnt, D_criterions, training_mode = "000"):
    """
    Args:
        agent:
        history_pool:
        batch: real data, fake data will generated from G
        sample_shape: is for D training
        dis_optimizer:
        batch_cnt: no use perhaps will add annealing function in the future.
        training_mode: for special training
    Returns: Nothing, just train D using history again.
    """
    for _ in range(1):
        dis_optimizer.zero_grad()
        if len(history_pool.experience_pool)<1:
            break
        fake_s_a = history_pool.next()
        disc_loss, _ = agent.disc_train(sample_shape, batch, D_criterions, training_mode, fake_s_a)
        # disc_loss, train_acc = agent.disc_train(sample_shape, batch)
        agent.discriminator.backward(batch_cnt, disc_loss)
        dis_optimizer.step()

def gan_train(agent, machine_data, train_feed, valid_feed, test_feed, config, evaluator, pred_list=[], gen_sampled_list=[]):
    """
    Args:
        agent: VAE, G and D
        machine_data: [state, action_randperm] : [state, action_noshuffle]
        train_feed:
        valid_feed:
        test_feed:
        config:
        evaluator:
        pred_list: prediction list is empty in the start.
        gen_sampled_list: empty list in the start

    Returns:

    """
    # specify the best loss
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    batch_cnt = 0
    vae_flag = config.vae_train
    dis_optimizer = agent.discriminator.get_optimizer(config)
    gen_optimizer = agent.generator.get_optimizer(config)
    # specify the loss function.
    D_criterions = []
    for l in range(3):
        D_criterions.append(torch.nn.BCELoss(reduction="sum"))
    # specify training model, 1 meaning real, and 0 meaning fake
    # training_modes = ["011", "101", "110", "001", "010", "100", "000"]

    """
    Hyperparamaters
    """
    vae_flag = False
    # training_modes = ["000", "011", "000", "101", "000","110"]
    training_modes = ["011", "101", "110", "001", "010", "100", "000"]
    config.max_epoch_list = [300, 300, 300, 300, 300, 300, 300]
    config.gan_ratio_g = 5

    if config.domain=='multiwoz' and vae_flag:
        gen_vae_optimizer = agent.gan_vae_optimizer(config)

    optimizer_com = torch.optim.Adam(agent.parameters(), lr=0)
    history_pool = HistoryData(10000)
    done_epoch = 0
    agent.train()

    logger.info("**** GAN Training Begins ****")
    disc_on_random_data, epoch_valid = [1.0, 1.0], 0
    # best_valid_loss = 100
    for idx, training_mode in enumerate(training_modes):
        largest_diff = -1
        patience = config.max_patience
        config.max_epoch = config.max_epoch_list[idx]
        logger.info("**** Epoch 0/{} For training mode: {} ****".format(config.max_epoch, training_mode))
        done_epoch = 0
        train_loss = LossManager()
        break_second = False

        # start from here
        while True:
            train_feed.epoch_init(config, verbose = done_epoch == 0, shuffle = True)
            batch_count_inside=-1
            # adjust_learning_rate(dis_optimizer, done_epoch, config)
            """
            Take Train Batch
                Train D on G
                Train D on history
                Train G
            Evaluation of VAL
            """
            # -------
            """
            dis_optimizer
            gen_optimizer
            gen_vae_optimizer : gen + vae
            """
            while True:
                if config.domain=='multiwoz' and vae_flag:
                    agent.vae.train()
                else:
                    agent.vae.eval()

                batch_count_inside+=1
                batch = train_feed.next_batch()
                sample_shape = config.batch_size, config.state_noise_dim, config.action_noise_dim
                if batch is None:
                    agent.discriminator.decay_noise()
                    break
                # ''''''''''''''' Training discriminator '''''''''''''''
                for _ in range(config.gan_ratio):
                    if config.gan_type=='wgan':
                        for p in agent.discriminator.parameters():
                            p.data.clamp_(-0.03, 0.03)
                    # for p in agent.discriminator.parameters():
                    #     p.data.clamp_(-0.03, 0.03)

                    dis_optimizer.zero_grad()
                    # this is for all models in agent.
                    optimizer_com.zero_grad()
                    # only feed the real data from training.
                    disc_loss, train_acc = agent.disc_train(sample_shape, batch, D_criterions, training_mode = training_mode)
                    agent.discriminator.backward(batch_cnt, disc_loss)
                    dis_optimizer.step()
                    train_disc_with_history(agent, history_pool, batch, sample_shape, dis_optimizer, batch_cnt, D_criterions, training_mode = training_mode)

                # ''''''''''''''' Training generator '''''''''''''''\
                for _ in range(config.gan_ratio_g):
                    if config.domain=='multiwoz' and vae_flag:
                        gen_vae_optimizer.zero_grad()
                    else:
                        gen_optimizer.zero_grad()

                    gen_loss, fake_s_a = agent.gen_train(batch, sample_shape, training_mode = training_mode)
                    agent.generator.backward(batch_cnt, gen_loss)

                    if config.domain=='multiwoz' and vae_flag:
                        gen_vae_optimizer.step()
                    else:
                        gen_optimizer.step()

                # only add once, no duplicate, okay?
                history_pool.add(fake_s_a)


                batch_cnt += 1
                train_loss.add_loss(disc_loss)

                # how many time to evaluate
                if batch_count_inside == 0 and done_epoch % 1==0:
                    logger.info("\n**** Epcoch {}/{} Done for Training Mode: {} ****".format(done_epoch, config.max_epoch, training_mode))
                    logger.info("\n====== Evaluating Model ======")
                    logger.info(train_loss.pprint("Train"))
                    # So this one is the bugs over here.
                    logger.info("Average disc value for human and machine on training set: {:.3f}, {:.3f}".format(train_acc[-2], train_acc[-1]))

                    # validation
                    agent.eval()
                    logger.info("====Validate Discriminator====")
                    valid_loss_disc         = disc_validate(agent, valid_feed, config, sample_shape, D_criterions, batch_cnt, training_mode = training_mode)
                    logger.info("======Validate Generator======")
                    valid_loss, gen_samples = gen_validate (agent, valid_feed, config, sample_shape, D_criterions, done_epoch, batch_cnt, training_mode = training_mode)
                    if len(gen_samples) > 0:
                        gen_sampled_list.append([done_epoch, gen_samples])
                    logger.info("====Validate Discriminator for t-SNE====")
                    if   agent.vae.vae_in_size == 392:
                        pred, disc_value = disc_validate_for_tsne_single_input(agent, machine_data, valid_feed, config, sample_shape)
                    elif agent.vae.vae_in_size == 492:
                        pred, disc_value = disc_validate_for_tsne_state_action_embed(agent, machine_data, valid_feed, config, sample_shape)
                    else:
                        raise ValueError("no such domain: {}".format(config.domain))
                    pred_list.append(pred)

                    if config.save_model:
                        disc_on_random_data = disc_value
                        epoch_valid = done_epoch
                        best_valid_loss = valid_loss

                        # we wll take the larget_diff as our evaluation of the mode (good or not), early stop.
                        if disc_value[0] - disc_value[1] >= largest_diff:
                            # set to the origional
                            largest_diff = (disc_value[0] - disc_value[1])
                            patience = config.max_patience
                            # save current model, since it going downer, using greedy method
                            save_model_woz(agent, config, training_mode = training_mode)
                            logger.info("--->>>Reset the patience")

                        else:
                            patience -= 1
                            logger.info("--->>>Still need {} times to end \n".format(patience))

                        # exit when meet the max_epoch | early_stop
                        if done_epoch >= config.max_epoch or config.early_stop and patience <= 0:
                            break_second = True
                            if done_epoch < config.max_epoch:
                                logger.info("Training_mode: "+ training_mode +"  !!Early stop due to run out of patience!!")
                            # at last to output the best validation loss.
                            # the best is the last one, not the best one in all of the training process.
                            logger.info("Best VAL_D %f" % largest_diff)
                            logger.info("Best validation loss %f" % best_valid_loss)
                            logger.info("Best validation Epoch on Machine data: {}".format(epoch_valid))
                            logger.info("Best validation Loss: {}".format(best_valid_loss))
                            logger.info("Best validation value on Machine data: {}, {}".format(disc_on_random_data[0], disc_on_random_data[1]))
                            # return
                            if training_mode == training_modes[-1]:
                                return
                            else:
                                break

                    # exit eval model
                    agent.train()
                    train_loss.clear()

            # break the second loop
            if break_second == True: break
            done_epoch += 1

def vae_train(agent, train_feed, valid_feed, test_feed, config):
    # patience stuff
    patience = config.vae_max_patience  # wait for at least 100 epoch before stop
    valid_loss_threshold = np.inf
    # Todo, add this stuff.
    best_valid_loss = [np.inf, np.inf]

    batch_cnt = 0
    vae_optimizer = agent.vae.get_optimizer(config)
    train_loss = LossManager()
    agent.vae.train()
    done_epoch = 0
    disc_on_random_data, epoch_valid = [1.0, 1.0], 0
    largest_diff = -1
    if config.vae: logger.info("----- VAE Training Begins -----")
    else: logger.info("----- AE Training Begins -----")
    logger.info("----- Epoch 0/{} -----".format(config.max_epoch))

    while True:
        train_feed.epoch_init(config, verbose = done_epoch == 0, shuffle=True)
        batch_count_inside = -1
        # adjust_learning_rate(dis_optimizer, done_epoch, config)

        while True:
            batch_count_inside += 1
            batch = train_feed.next_batch()
            if batch is None:
                break
            vae_optimizer.zero_grad()
            vae_loss= agent.vae_train(batch, done_epoch)
            agent.vae.backward(batch_cnt, vae_loss)
            vae_optimizer.step()

            batch_cnt += 1
            train_loss.add_loss(vae_loss)
            # train_loss.add_loss(disc_loss_self)

            if batch_count_inside==0:
                logger.info("\n---------- AutoEncoder Epoch: {} % {} ----------".format(done_epoch, config.max_epoch))
                logger.info(train_loss.pprint("Train"))
                valid_loss, kl_loss_whole = vae_validate(agent, valid_feed, config, batch_cnt)
                # save_model_vae(agent, config)
                # update early stopping stats
                # forget about KL_Loss, only use the valid loss to compute this stuff.
                if (valid_loss < best_valid_loss[0]) and done_epoch > 5:
                    save_model_vae(agent, config)
                    best_valid_loss[0] = min(   valid_loss, best_valid_loss[0])
                    best_valid_loss[1] = min(kl_loss_whole, best_valid_loss[1])
                    patience = config.vae_max_patience
                    logger.info("--->>>Reset the patience, current min is:{:.3f} and {:.3f} ".format(best_valid_loss[0], best_valid_loss[1]))


                else:
                    patience -= 1
                    logger.info("--->>>Still need {} times to end \n".format(patience))
                    # save model every 5 epoch
                    if done_epoch % 5 == 0:
                        save_model_vae(agent, config, done_epoch)

                if (done_epoch >= config.max_epoch) or (config.early_stop and patience <= 0):
                    if done_epoch < config.max_epoch:
                        logger.info("!!Early stop due to run out of patience!!")
                    test_loss = vae_validate(agent, test_feed, config, batch_cnt)
                    logger.info("Best val Loss: [{}, {}], Best test loss".format(best_valid_loss[0], best_valid_loss[1], test_loss))
                    return True
                # exit eval model
                agent.train()
                train_loss.clear()
                
        done_epoch += 1

        # KL is still not running out.