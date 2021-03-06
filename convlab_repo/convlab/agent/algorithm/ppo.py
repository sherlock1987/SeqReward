# Modified by Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy

import numpy as np
import torch
import copy
import os
import argparse
import json

from collections import deque
from convlab.agent.algorithm import policy_util
from convlab.agent.algorithm.actor_critic import ActorCritic
from convlab.agent.net import net_util
from convlab.lib import logger, math_util, util
from convlab.lib.decorator import lab_api
from convlab.agent.algorithm import reward_agent, reward_utils
logger = logger.get_logger(__name__)


class PPO(ActorCritic):
    '''
    Implementation of PPO
    This is actually just ActorCritic with a custom loss function
    Original paper: "Proximal Policy Optimization Algorithms"
    https://arxiv.org/pdf/1707.06347.pdf

    Adapted from OpenAI baselines, CPU version https://github.com/openai/baselines/tree/master/baselines/ppo1
    Algorithm:
    for iteration = 1, 2, 3, ... do
        for actor = 1, 2, 3, ..., N do
            run policy pi_old in env for T timesteps
            compute advantage A_1, ..., A_T
        end for
        optimize surrogate L wrt theta, with K epochs and minibatch size M <= NT
    end for

    e.g. algorithm_spec
    "algorithm": {
        "name": "PPO",
        "action_pdtype": "default",
        "action_policy": "default",
        "explore_var_spec": null,
        "gamma": 0.99,
        "lam": 1.0,
        "clip_eps_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
        "entropy_coef_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
        "minibatch_size": 256,
        "training_frequency": 1,
        "training_epoch": 8,
    }

    e.g. special net_spec param "shared" to share/separate Actor/Critic
    "net": {
        "type": "MLPNet",
        "shared": true,
        ...
    '''

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            explore_var_spec=None,
            entropy_coef_spec=None,
            minibatch_size=4,
            val_loss_coef=1.0,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # theoretically, PPO does not have policy update; but in this implementation we have such option
            'explore_var_spec',
            'gamma',
            'lam',
            'clip_eps_spec',
            'entropy_coef_spec',
            'val_loss_coef',
            'minibatch_size',
            'training_frequency',  # horizon
            'training_epoch',
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.explore_var_scheduler = policy_util.VarScheduler(self.explore_var_spec)
        self.body.explore_var = self.explore_var_scheduler.start_val
        # extra variable decays for PPO
        self.clip_eps_scheduler = policy_util.VarScheduler(self.clip_eps_spec)
        self.body.clip_eps = self.clip_eps_scheduler.start_val
        if self.entropy_coef_spec is not None:
            self.entropy_coef_scheduler = policy_util.VarScheduler(self.entropy_coef_spec)
            self.body.entropy_coef = self.entropy_coef_scheduler.start_val
        # PPO uses GAE
        self.calc_advs_v_targets = self.calc_gae_advs_v_targets

    @lab_api
    def init_nets(self, global_nets=None):
        '''PPO uses old and new to calculate ratio for loss'''
        super().init_nets(global_nets)
        # create old net to calculate ratio
        self.old_net = deepcopy(self.net)
        assert id(self.old_net) != id(self.net)
    
        val_feed = reward_utils.WoZGanDataLoaders('val', 64)
        train_feed = reward_utils.WoZGanDataLoaders('train', 64)
        train_feed.epoch_init(shuffle=True)
        
        use_gpu = False
        vae_type = 'vae'
        update = False

        self.experience_buffer = deque(maxlen=10)
        # len was 200 at the beginnning
        self.reward_buffer = deque(maxlen=self.algorithm_spec['reward_buffer_size'])

        """
        my reward model start from here.
        Just change the reward_path and the function is enough for me.
        Potential bugs in actot critic, since this one is the basic function for PPO.
        """
        # ziming's code
        # self.reward_agent = reward_agent.RewardAgent_EncoderSide(use_gpu, vae_type,update=update, real_data_feed=train_feed)   # this is the State Vae and Action Onehot version
        # reward_path = 'convlab_repo/saved_models/2019-09-06T12:04:49.278628-mwoz_gan_vae.py' # the pre trained vae-based reward
        root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        # reward_path = os.path.join(root_dir, "convlab_repo/saved_models/naive_v_parallel_cl")
        # reward_path = os.path.join(root_dir, "convlab_repo/saved_models/cl_2_VAE")
        # reward_path = os.path.join(root_dir, "convlab_repo/saved_models/cl_3_VAE_no_kl_finish")
        reward_path = os.path.join(root_dir, "convlab_repo/saved_models/cl_3_VAE_pre_training_mode")

        config_path = os.path.join(reward_path, "params.json")
        with open(config_path, 'r') as f:
            dic = json.load(f)
            config = argparse.Namespace(**dic)
        self.reward_agent = reward_agent.RewardAgent_EncoderSide(config, use_gpu, "mine", vae_type)


        self.optim_gandisc=None
        # no update for the ppo.
        if update:
            self.optim_gandisc = self.reward_agent.discriminator.get_optimizer(config)
        

        self.disc_training_times = self.algorithm_spec['disc_training_times']
        self.disc_training_freq = self.algorithm_spec['disc_training_freq']
        # self.reward_type = self.algorithm_spec['reward_type']
        # self.reward_type = 'AIRL'
        # self.reward_type = 'DISC'
        self.reward_type = 'OFFGAN'
        # self.reward_type = 'OFFGAN_update'
        # self.reward_type = 'Human'
        # self.reward_type = "my_parallel"

        if self.reward_type=='DISC':
            self.discriminator = reward_agent.A2C_Discriminator(config, use_gpu, train_feed, 64)
            disc_mdl = './reward_model/disc_pretrain.mdl'  
        else:
            self.discriminator = reward_agent.AIRL(config, use_gpu, train_feed, 64)
            disc_mdl = './reward_model/airl_pretrain.mdl'
            # if os.path.exists(disc_mdl):
            #     self.discriminator.load_state_dict(torch.load(disc_mdl))
            #     print("successfully loaded the pretrained Disc model")
        self.optim_disc = self.discriminator.get_optimizer()
        self.disc_training_count = 0
        self.policy_training_flag=False

        # load model
        reward_agent.load_reward_model(self.reward_agent, reward_path, use_gpu)
        if use_gpu:
            self.reward_agent.cuda()
        self.reward_agent.eval()
        self.reward_count = 0
        self.batch_count = 0
        self.pretrain_finished = False
        self.pretrain_disc_and_valud_finished = False
        self.disc_pretrain_finished = False
        if self.reward_type=='OFFGAN':
            self.disc_pretrain_finished = True
            self.policy_training_flag=True
            self.pretrain_finished = False
 

        # reward_utils.reward_validate(self.reward_agent, val_feed)
        self.load_pretrain_policy = self.algorithm_spec['load_pretrain_policy']
        policy_mdl = './reward_model/policy_pretrain.mdl'

        if self.load_pretrain_policy:
            if os.path.exists(policy_mdl):
                self.net.load_state_dict(torch.load(policy_mdl))
                self.old_net.load_state_dict(torch.load(policy_mdl))
                print("successfully loaded the pretrained policy model")
            else:
                raise ValueError("No policy model")


    def calc_policy_loss(self, batch, pdparams, advs):
        '''
        The PPO loss function (subscript t is omitted)
        L^{CLIP+VF+S} = E[ L^CLIP - c1 * L^VF + c2 * S[pi](s) ]

        Breakdown piecewise,
        1. L^CLIP = E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]
        where ratio = pi(a|s) / pi_old(a|s)

        2. L^VF = E[ mse(V(s_t), V^target) ]

        3. S = E[ entropy ]
        '''
        clip_eps = self.body.clip_eps
        action_pd = policy_util.init_action_pd(self.body.ActionPD, pdparams)
        states = batch['states']
        actions = batch['actions']
        if self.body.env.is_venv:
            states = math_util.venv_unpack(states)
            actions = math_util.venv_unpack(actions)

        # L^CLIP
        log_probs = action_pd.log_prob(actions)
        with torch.no_grad():
            old_pdparams = self.calc_pdparam(states, net=self.old_net)
            old_action_pd = policy_util.init_action_pd(self.body.ActionPD, old_pdparams)
            old_log_probs = old_action_pd.log_prob(actions)
        assert log_probs.shape == old_log_probs.shape
        ratios = torch.exp(log_probs - old_log_probs)  # clip to prevent overflow
        logger.debug(f'ratios: {ratios}')
        sur_1 = ratios * advs
        sur_2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advs
        # flip sign because need to maximize
        clip_loss = -torch.min(sur_1, sur_2).mean()
        logger.debug(f'clip_loss: {clip_loss}')

        # L^VF (inherit from ActorCritic)

        # S entropy bonus
        entropy = action_pd.entropy().mean()
        self.body.mean_entropy = entropy  # update logging variable
        ent_penalty = -self.body.entropy_coef * entropy
        logger.debug(f'ent_penalty: {ent_penalty}')

        policy_loss = clip_loss + ent_penalty
        logger.debug(f'PPO Actor policy loss: {policy_loss:g}')
        return policy_loss

    def fetch_irl_reward(self, batch):
        self.reward_agent.eval()
        with torch.no_grad():
            """
            Change the surgery over here.
            """
            reward = self.reward_agent(copy.deepcopy(batch), surgery = "hard").detach().view(-1)
        # if len(self.reward_buffer)>5:
        #     reward = reward - sum(self.reward_buffer)/len(self.reward_buffer)
        reward_log = torch.log(reward)
        reward_log_double = torch.log(reward) - torch.log(1 - reward)
        reward_log_minus_one = - torch.log(1 - reward)
        return reward, reward_log, reward_log_double, reward_log_minus_one
    
    def replace_reward_batch(self, batch):
        batch_rewards_ori, batch_rewards_log, batch_rewards_log_double, batch_reward_log_minus_one = self.fetch_irl_reward(batch)
        self.reward_count += batch_rewards_ori.mean().item()
        self.reward_buffer.append(batch_rewards_log.mean().item())
        self.batch_count += 1
        # print("average gan rewards: {}".format(batch_rewards_ori.mean().item()))
        # batch['rewards'] = batch_rewards_ori + batch['rewards'] 
        if len(self.reward_buffer)>3:  # stabilize
            batch['rewards'] = batch_rewards_log - sum(self.reward_buffer)/len(self.reward_buffer)+ batch['rewards']
        else:
            batch['rewards'] = batch_rewards_log + batch['rewards']
        return batch
    
    def fetch_offgan_reward(self, batch):
        batch_rewards_ori, batch_rewards_log, batch_rewards_log_double, batch_reward_log_minus_one = self.fetch_irl_reward(batch)
        self.reward_count += batch_rewards_ori.mean().item()
        self.batch_count += 1
        print("average gan rewards: {}".format(batch_rewards_ori.mean().item())) 
        batch['rewards'] = batch_rewards_log_double

        # if not self.pretrain_finished:
            # return batch
        self.experience_buffer.append(copy.deepcopy(batch))
        self.reward_agent.discriminator.train()
        self.GanReward_Update(4)
        self.reward_agent.discriminator.eval()
        return batch

    def fetch_disc_reward(self, batch):
        self.disc_training_count +=1
        if self.disc_training_count>=self.disc_training_freq:
            self.policy_training_flag=True
            self.disc_training_count = 0
 
        if not self.disc_pretrain_finished or self.pretrain_finished:
            self.experience_buffer.append(copy.deepcopy(batch))
            self.discriminator.train()
            self.disc_train(self.disc_training_times)
            self.discriminator.eval()

        self.discriminator.eval()
        disc_r = self.discriminator.get_reward(batch).view(-1)
        print('Disc reward: {}'.format(disc_r.mean().item()))
        batch['rewards'] = disc_r 


        self.batch_count += 1
        return batch

    def fetch_airl_reward(self, batch):
        self.disc_training_count +=1
        if self.disc_training_count>=self.disc_training_freq:
            self.policy_training_flag=True
            self.disc_training_count = 0

        if not self.disc_pretrain_finished or self.pretrain_finished:
            self.experience_buffer.append(copy.deepcopy(batch))
            self.discriminator.train()
            self.airl_train(self.disc_training_times)
            self.discriminator.eval()

        self.discriminator.eval()
        pdparams, _ = self.calc_pdparam_v(batch)
        action_pd = policy_util.init_action_pd(self.body.ActionPD, pdparams)
        actions = batch['actions']
        log_probs = action_pd.log_prob(actions)
        weight = self.discriminator.get_reward(batch)
        assert log_probs.shape == weight.shape
        reward = (weight - log_probs.view(-1)).detach()

        print('airl weight: {}, airl logp: {}, airl reward: {}'.format(weight.mean().item(), log_probs.mean().item(), reward.mean().item()))


        batch['rewards'] = reward


        self.batch_count += 1
        return batch


    def imitate_loop(self):
        real_state, real_action = self.discriminator.sample_real_batch_id()
        batch = {'states':real_state, 'actions':real_action}
        pdparams, _ = self.calc_pdparam_v(batch)
        action_pd = policy_util.init_action_pd(self.body.ActionPD, pdparams)
        log_probs = action_pd.log_prob(real_action)
        imitate_loss = -log_probs.mean()

        return imitate_loss

    def imitate_train(self, times=500):
        clock = self.body.env.clock
        loss = 0.
        for _ in range(times):
            imitate_loss = self.imitate_loop()
            self.net.train_step(imitate_loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
            loss += imitate_loss.item()
        return imitate_loss


    def pretrain(self):
        loss = self.imitate_train()
        return loss.detach().numpy()


    def train(self):
        # torch.save(self.net.state_dict(), './reward_model/policy_pretrain.mdl')
        # raise ValueError("policy pretrain stops")
        if util.in_eval_lab_modes():
            return np.nan
        clock = self.body.env.clock

        if self.body.env.clock.epi > 200:   # this is for the adversarial training if you have a discriminator to train.
            self.disc_pretrain_finished = True
        if self.body.env.clock.epi > 200:
            self.pretrain_finished = True
            # torch.save(self.discriminator.state_dict(), './reward_model/airl_pretrain.mdl')
            # raise ValueError("pretrain stops here")
            """
            Specify Surgery. 
            Change the reward path.
            That's it.
            """

        if self.to_train == 1:
            net_util.copy(self.net, self.old_net)  # update old net
            batch = self.sample()
            if self.reward_type =='OFFGAN':
                batch = self.replace_reward_batch(batch)
            # if self.reward_type =='DISC':
            #     batch = self.fetch_disc_reward(batch)
            # if self.reward_type =='AIRL':
            #     batch = self.fetch_airl_reward(batch)
            # if self.reward_type == 'OFFGAN_update':
            #     batch = self.fetch_offgan_reward(batch)
            
            clock.set_batch_size(len(batch))
            _pdparams, v_preds = self.calc_pdparam_v(batch)
            advs, v_targets = self.calc_advs_v_targets(batch, v_preds)
            # piggy back on batch, but remember to not pack or unpack
            batch['advs'], batch['v_targets'] = advs, v_targets
            if self.body.env.is_venv:  # unpack if venv for minibatch sampling
                for k, v in batch.items():
                    if k not in ('advs', 'v_targets'):
                        batch[k] = math_util.venv_unpack(v)
            total_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                minibatches = util.split_minibatch(batch, self.minibatch_size)

                # if not self.pretrain_finished or not self.policy_training_flag:
                #     break

                for minibatch in minibatches:
                    if self.body.env.is_venv:  # re-pack to restore proper shape
                        for k, v in minibatch.items():
                            if k not in ('advs', 'v_targets'):
                                minibatch[k] = math_util.venv_pack(v, self.body.env.num_envs)
                    advs, v_targets = minibatch['advs'], minibatch['v_targets']
                    pdparams, v_preds = self.calc_pdparam_v(minibatch)
                    policy_loss = self.calc_policy_loss(minibatch, pdparams, advs)  # from actor
                    val_loss = self.calc_val_loss(v_preds, v_targets)  # from critic
                    if self.shared:  # shared network
                        loss = policy_loss + val_loss
                        self.net.train_step(loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
                    else:
                        # pretrain_finished = false -> policy keep fixed, updating value net and disc
                        # if not self.pretrain_finished and self.disc_pretrain_finished:
                        #     self.critic_net.train_step(val_loss, self.critic_optim, self.critic_lr_scheduler, clock=clock, global_net=self.global_critic_net)
                        #     loss = val_loss
                        # elif self.pretrain_finished and self.policy_training_flag:
                        self.net.train_step(policy_loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
                        self.critic_net.train_step(val_loss, self.critic_optim, self.critic_lr_scheduler, clock=clock, global_net=self.global_critic_net)
                        loss = policy_loss + val_loss
                            # _ = self.imitate_train(10)
                    total_loss += loss
            loss = total_loss / self.training_epoch / len(minibatches)
            if not self.pretrain_finished:
                logger.info("warmup Value net, epi: {}, frame: {}, loss: {}".format(clock.epi, clock.frame, loss))
            # reset
            self.to_train = 0
            # self.policy_training_flag=False   # this is for adversarial training  
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, frame: {clock.frame}, t: {clock.t}, total_reward so far: {self.body.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    def disc_train(self, training_times=1):
        for t in range(training_times):
            # idx = min(t+1, len(self.experience_buffer))
            batch = self.experience_buffer[-1]
            minibatches = util.split_minibatch(batch, 64)
            for fake_batch in minibatches:
                self.optim_disc.zero_grad()
                loss = self.discriminator.disc_train(fake_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 3)
                self.optim_disc.step()
    
    def airl_train(self, training_times=1):
        # print("airl training")
        for t in range(training_times):
            total_loss = 0
            # idx = min(t+1, len(self.experience_buffer))
            batch = self.experience_buffer[-1]
            minibatches = util.split_minibatch(batch, 64)
            # print("minibatch number: {}".format(len(minibatches)))
            for fake_batch in minibatches:
                self.optim_disc.zero_grad()
                loss = self.discriminator.disc_train(fake_batch)
                total_loss += loss.item()
                loss.backward()
                self.optim_disc.step()
                for p in self.discriminator.parameters():
                    p.data.clamp_(-0.1, 0.1)
            logger.info("airl training loss: {}".format(total_loss/len(minibatches)))

    def GanReward_Update(self, training_times=1):
        for _ in range(training_times):
            batch = self.experience_buffer[-1]
            minibatches = util.split_minibatch(batch, 64)
            for fake_batch in minibatches:
                loss = self.reward_agent.update(fake_batch)
                self.optim_gandisc.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.reward_agent.discriminator.parameters(), 0.5)
                self.optim_gandisc.step()


    def value_train(self):
        clock = self.body.env.clock
        batch = self.sample()
        print("batch size: {}".format(len(batch['states'])))
        _pdparams, v_preds = self.calc_pdparam_v(batch)
        advs, v_targets = self.calc_advs_v_targets(batch, v_preds)
        val_loss = self.calc_val_loss(v_preds, v_targets)
        self.net.train_step(val_loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
        return val_loss.detach().numpy()


    @lab_api
    def update(self):
        self.body.explore_var = self.explore_var_scheduler.update(self, self.body.env.clock)
        if self.entropy_coef_spec is not None:
            self.body.entropy_coef = self.entropy_coef_scheduler.update(self, self.body.env.clock)
        self.body.clip_eps = self.clip_eps_scheduler.update(self, self.body.env.clock)
        return self.body.explore_var
