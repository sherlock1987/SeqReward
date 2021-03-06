# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
print(root_dir)

import numpy as np
import torch
import copy
import argparse
import json

from convlab.agent import memory
from convlab.agent import net
from convlab.agent.algorithm.sarsa import SARSA
from convlab.agent.net import net_util
from convlab.lib import logger, util
from convlab.lib.decorator import lab_api
from convlab.agent.algorithm import reward_agent, reward_utils
logger = logger.get_logger(__name__)



# No use
class VanillaDQN(SARSA):
    '''
    Implementation of a simple DQN algorithm.
    Algorithm:
        1. Collect some examples by acting in the environment and store them in a replay memory
        2. Every K steps sample N examples from replay memory
        3. For each example calculate the target (bootstrapped estimate of the discounted value of the state and action taken), y, using a neural network to approximate the Q function. s' is the next state following the action actually taken.
                y_t = r_t + gamma * argmax_a Q(s_t', a)
        4. For each example calculate the current estimate of the discounted value of the state and action taken
                x_t = Q(s_t, a_t)
        5. Calculate L(x, y) where L is a regression loss (eg. mse)
        6. Calculate the gradient of L with respect to all the parameters in the network and update the network parameters using the gradient
        7. Repeat steps 3 - 6 M times
        8. Repeat steps 2 - 7 Z times
        9. Repeat steps 1 - 8

    For more information on Q-Learning see Sergey Levine's lectures 6 and 7 from CS294-112 Fall 2017
    https://www.youtube.com/playlist?list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3

    e.g. algorithm_spec
    "algorithm": {
        "name": "VanillaDQN",
        "action_pdtype": "Argmax",
        "action_policy": "epsilon_greedy",
        "explore_var_spec": {
            "name": "linear_decay",
            "start_val": 1.0,
            "end_val": 0.1,
            "start_step": 10,
            "end_step": 1000,
        },
        "gamma": 0.99,
        "training_batch_iter": 8,
        "training_iter": 4,
        "training_frequency": 10,
        "training_start_step": 10,
    }
    '''

    @lab_api
    def init_algorithm_params(self):
        # set default
        util.set_attr(self, dict(
            action_pdtype='Argmax',
            action_policy='epsilon_greedy',
            explore_var_spec=None,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # explore_var is epsilon, tau or etc. depending on the action policy
            # these control the trade off between exploration and exploitaton
            'explore_var_spec',
            'gamma',  # the discount factor
            'training_batch_iter',  # how many gradient updates per batch
            'training_iter',  # how many batches to train each time
            'training_frequency',  # how often to train (once a few timesteps)
            'training_start_step',  # how long before starting training
        ])
        super().init_algorithm_params()

    @lab_api
    def init_nets(self, global_nets=None):
        '''Initialize the neural network used to learn the Q function from the spec'''
        if self.algorithm_spec['name'] == 'VanillaDQN':
            assert all(k not in self.net_spec for k in ['update_type', 'update_frequency', 'polyak_coef']), 'Network update not available for VanillaDQN; use DQN.'
        in_dim = self.body.state_dim
        out_dim = net_util.get_out_dim(self.body)
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, in_dim, out_dim)
        self.net_names = ['net']
        # init net optimizer and its lr scheduler
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self.optim, self.net.lr_scheduler_spec)
        net_util.set_global_nets(self, global_nets)
        self.post_init_nets()


    def calc_q_loss(self, batch):
        '''Compute the Q value loss using predicted and target Q values from the appropriate networks'''
        states = batch['states']
        next_states = batch['next_states']
        q_preds = self.net(states)
        with torch.no_grad():
            next_q_preds = self.net(next_states)
        act_q_preds = q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)).squeeze(-1)
        # Bellman equation: compute max_q_targets using reward and max estimated Q values (0 if no next_state)
        max_next_q_preds, _ = next_q_preds.max(dim=-1, keepdim=True)
        max_q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * max_next_q_preds
        logger.debug(f'act_q_preds: {act_q_preds}\nmax_q_targets: {max_q_targets}')
        q_loss = self.net.loss_fn(act_q_preds, max_q_targets)

        # TODO use the same loss_fn but do not reduce yet
        if 'Prioritized' in util.get_class_name(self.body.memory):  # PER
            errors = (max_q_targets - act_q_preds.detach()).abs().cpu().numpy()
            self.body.memory.update_priorities(errors)
        return q_loss

    @lab_api
    def act(self, state):
        '''Selects and returns a discrete action for body using the action policy'''
        return super().act(state)

    @lab_api
    def sample(self):
        '''Samples a batch from memory of size self.memory_spec['batch_size']'''
        batch = self.body.memory.sample()
        batch = util.to_torch_batch(batch, self.net.device, self.body.memory.is_episodic)
        return batch

    @lab_api
    def train(self):
        '''
        Completes one training step for the agent if it is time to train.
        i.e. the environment timestep is greater than the minimum training timestep and a multiple of the training_frequency.
        Each training step consists of sampling n batches from the agent's memory.
        For each of the batches, the target Q values (q_targets) are computed and a single training step is taken k times
        Otherwise this function does nothing.
        '''
        if util.in_eval_lab_modes():
            return np.nan
        clock = self.body.env.clock
        if self.to_train == 1:
            total_loss = torch.tensor(0.0)
            for _ in range(self.training_iter):
                batch = self.sample()
                clock.set_batch_size(len(batch))
                for _ in range(self.training_batch_iter):
                    loss = self.calc_q_loss(batch)
                    self.net.train_step(loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
                    total_loss += loss
            loss = total_loss / (self.training_iter * self.training_batch_iter)
            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, frame: {clock.frame}, t: {clock.t}, total_reward so far: {self.body.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    @lab_api
    def update(self):
        '''Update the agent after training'''
        return super().update()

# The base model, but not load reward over here.
class DQNBase(VanillaDQN):
    '''
    Implementation of the base DQN algorithm.
    The algorithm follows the same general approach as VanillaDQN but is more general since it allows
    for two different networks (through self.net and self.target_net).

    self.net is used to act, and is the network trained.
    self.target_net is used to estimate the maximum value of the Q-function in the next state when calculating the target (see VanillaDQN comments).
    self.target_net is updated periodically to either match self.net (self.net.update_type = "replace") or to be a weighted average of self.net and the previous self.target_net (self.net.update_type = "polyak")
    If desired, self.target_net can be updated slowly, and this can help to stabilize learning.

    It also allows for different nets to be used to select the action in the next state and to evaluate the value of that action through self.online_net and self.eval_net. This can help reduce the tendency of DQN's to overestimate the value of the Q-function. Following this approach leads to the DoubleDQN algorithm.

    Setting all nets to self.net reduces to the VanillaDQN case.
    '''

    @lab_api
    def init_nets(self, global_nets=None):
        '''Initialize networks'''
        if self.algorithm_spec['name'] == 'DQNBase':
            assert all(k not in self.net_spec for k in ['update_type', 'update_frequency', 'polyak_coef']), 'Network update not available for DQNBase; use DQN.'
        in_dim = self.body.state_dim
        out_dim = net_util.get_out_dim(self.body)
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, in_dim, out_dim)
        self.target_net = NetClass(self.net_spec, in_dim, out_dim)
        self.net_names = ['net', 'target_net']
        # init net optimizer and its lr scheduler
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self.optim, self.net.lr_scheduler_spec)
        net_util.set_global_nets(self, global_nets)
        self.post_init_nets()
        self.online_net = self.target_net
        self.eval_net = self.target_net

    def calc_q_loss(self, batch):
        '''Compute the Q value loss using predicted and target Q values from the appropriate networks'''
        states = batch['states']
        next_states = batch['next_states']
        q_preds = self.net(states)
        with torch.no_grad():
            # Use online_net to select actions in next state
            online_next_q_preds = self.online_net(next_states)
            # Use eval_net to calculate next_q_preds for actions chosen by online_net
            next_q_preds = self.eval_net(next_states)
        act_q_preds = q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)).squeeze(-1)
        online_actions = online_next_q_preds.argmax(dim=-1, keepdim=True)
        max_next_q_preds = next_q_preds.gather(-1, online_actions).squeeze(-1)
        max_q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * max_next_q_preds
        logger.debug(f'act_q_preds: {act_q_preds}\nmax_q_targets: {max_q_targets}')
        q_loss = self.net.loss_fn(act_q_preds, max_q_targets)

        # TODO use the same loss_fn but do not reduce yet
        if 'Prioritized' in util.get_class_name(self.body.memory):  # PER
            errors = (max_q_targets - act_q_preds.detach()).abs().cpu().numpy()
            self.body.memory.update_priorities(errors)
        return q_loss

    def update_nets(self):
        if util.frame_mod(self.body.env.clock.frame, self.net.update_frequency, self.body.env.num_envs):
            if self.net.update_type == 'replace':
                net_util.copy(self.net, self.target_net)
            elif self.net.update_type == 'polyak':
                net_util.polyak_update(self.net, self.target_net, self.net.polyak_coef)
            else:
                raise ValueError('Unknown net.update_type. Should be "replace" or "polyak". Exiting.')

    @lab_api
    def update(self):
        '''Updates self.target_net and the explore variables'''
        self.update_nets()
        return super().update()

# He add the reward model over here, not in the whole graph part, will this be the potential diff?
class DQN(DQNBase):
    '''
    DQN class
    e.g. algorithm_spec
    "algorithm": {
        "name": "DQN",
        "action_pdtype": "Argmax",
        "action_policy": "epsilon_greedy",
        "explore_var_spec": {
            "name": "linear_decay",
            "start_val": 1.0,
            "end_val": 0.1,
            "start_step": 10,
            "end_step": 1000,
        },
        "gamma": 0.99,
        "training_batch_iter": 8,
        "training_iter": 4,
        "training_frequency": 10,
        "training_start_step": 10
    }
    '''
    def __init__(self, agent, global_nets=None):
        super().__init__(agent, global_nets)
        use_gpu = False
        # vae_type = 'autoencoder'
        vae_type = 'vae'
        """
        1, 2 = 3
        """
        # self.reward_agent = reward_agent.RewardAgent_EncoderSide(use_gpu, vae_type)   # this is the State Vae and Action Onehot version
        # reward_path = './your/trained/model/path/2019-08-16T12-04-13-mwoz_gan_vae.py'  # r4  # this is the autoencoder based reward model
        """
        1
        """
        # reward_path = os.path.join(root_dir, "gan_v/logs/cl_1_AE")
        # reward_path = os.path.join(root_dir, "gan_v/logs/naive_model_1_vae_update")
        # reward_path = os.path.join(root_dir, "gan_v/logs/cl_1_AE_action_noise")
        # reward_path = os.path.join(root_dir, "convlab_repo/saved_models/naive_v_parallel_cl")
        # reward_path = os.path.join(root_dir, "convlab_repo/saved_models/cl_2_VAE")
        # reward_path = os.path.join(root_dir, "convlab_repo/saved_models/cl_3_VAE_no_kl_finish")
        # reward_path = os.path.join(root_dir, "convlab_repo/saved_models/cl_2_VAE")
        reward_path = os.path.join(root_dir, "convlab_repo/saved_models/cl_3_VAE_pre_training_mode")

        # abalation test model
        # reward_path = os.path.join(root_dir, "convlab_repo/saved_models/cl_finish_no_noise")
        config_path = os.path.join(reward_path, "params.json")
        with open(config_path, 'r') as f:
            dic = json.load(f)
            config = argparse.Namespace(**dic)
        self.reward_agent = reward_agent.RewardAgent_EncoderSide(config, use_gpu, model_name = "mine")   # this is the State Vae and Action Onehot version
        # reward_path = os.path.join(root_dir, "convlab_repo/saved_models/2019-09-06T12:04:49.278628-mwoz_gan_vae.py")
        """
        2, the normal one.
        """
        # self.reward_agent = reward_agent.RewardAgent_StateVaeActionSeg(use_gpu, vae_type)   # this is the State Vae and Action Seg version
        # reward_path = os.path.join(root_dir, 'convlab_repo/saved_models/2019-09-18T20:06:28.509357-mwoz_gan_vae_StateActionEmbed.py') # new trained state_vae action_seg reward, Hotel excluded
        """
        # 3
        # self.reward_agent = reward_agent.RewardAgent_StateVaeActionSeg(use_gpu, vae_type)   # this is the State Vae and Action Seg version
        # reward_path = os.path.join(root_dir, 'convlab_repo/saved_models/2019-09-19T22:06:56.826004-mwoz_gan_vae_StateActionEmbed.py') # new trained state_vae action_seg reward, All domains
        """
        reward_agent.load_reward_model(self.reward_agent, reward_path, use_gpu)
        if use_gpu:
            self.reward_agent.cuda()
        self.reward_agent.eval()
        self.reward_count = 0
        self.batch_count = 0

        # val_feed = reward_utils.WoZGanDataLoaders('val')
        # reward_utils.reward_validate(self.reward_agent, val_feed)

        """
        #DRAW of current stragetory.
        # For second agent loaded
        """
        reward_path_ziming = os.path.join(root_dir, "convlab_repo/saved_models/2019-09-06T12:04:49.278628-mwoz_gan_vae.py")
        ziming_agent = reward_agent.RewardAgent_EncoderSide(config, use_gpu, model_name = "ziming")   # this is the State Vae and Action Onehot version
        reward_agent.load_reward_model(ziming_agent, reward_path_ziming, use_gpu)

        test_feed = reward_utils.WoZGanDataLoaders("test")
        reward_utils.plot_graph(self.reward_agent, test_feed, surgery = "das")
        reward_utils.plot_graph(self.reward_agent, test_feed, surgery = "hard_update")
        # reward_utils.plot_graph(self.reward_agent, test_feed, surgery = "product")
        reward_utils.plot_graph(ziming_agent, test_feed, name = "ziming")
        import random
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        random.seed(1)
        np.random.seed(1)
        # Do a through evaluation.
        type_list = ["hard_update", "das", "product"]
        # type_list = ["d", "a", "s"]        #

        reward_utils.plot_graph_4_seperate(self.reward_agent, ziming_agent, test_feed, type_list)
        # reward_utils.plot_graph_4_seperate(self.reward_agent, ziming_agent, test_feed, type_list)
        # type_list = ["d", "a", "s"]        #
        # reward_utils.plot_graph_4_seperate(self.reward_agent, ziming_agent, test_feed, type_list)

    @lab_api
    def init_nets(self, global_nets=None):
        super().init_nets(global_nets)
    
    # def fetch_irl_reward_batch(self, batch):
    #     reward = self.reward_agent(copy.deepcopy(batch))
    #     batch['irl_rewards'] = reward
    #     return batch

    def fetch_irl_reward(self, batch, surgery = "das"):
        """
        Args:
            batch:
        Returns: four types of reward function.
        """
        self.reward_agent.eval()
        with torch.no_grad():
            reward = self.reward_agent(copy.deepcopy(batch), surgery = surgery).detach().view(-1)
        reward_log = torch.log(reward)
        reward_log_double = torch.log(reward) - torch.log(1 - reward)
        reward_log_minus_one = - torch.log(1 - reward)
        return reward, reward_log, reward_log_double, reward_log_minus_one

    def calc_q_loss(self, batch):
        '''Compute the Q value loss using predicted and target Q values from the appropriate networks'''
        batch_rewards_ori, batch_rewards_log, batch_rewards_log_double, batch_reward_log_minus_one = self.fetch_irl_reward(batch)
        self.reward_count += batch_rewards_ori.mean().item()
        self.batch_count += 1
        # batch_rewards = batch_rewards_ori + batch['rewards']
        # batch_rewards = batch_reward_log_minus_one
        # batch_rewards = batch_rewards_log
        # batch_rewards = batch_rewards_log_double + batch['rewards']
        """
        here to change the reward function. From two to choose one. For me, baseline is not running over here.
        Specify the method of surgery
        change VAE function in the other place.
        """
        batch_rewards = batch_rewards_log.to("cpu") + batch['rewards']
        # batch_rewards = batch['rewards']
        # batch_rewards = batch_rewards_ori.to("cpu") + batch['rewards']
        # flag = copy.deepcopy(batch['rewards'])
        # flag[flag<=0]=0
        # flag[flag>0]=1
        # batch_rewards = batch_rewards_log + flag * batch['rewards']

        states = batch['states']
        next_states = batch['next_states']
        q_preds = self.net(states)
        with torch.no_grad():
            # Use online_net to select actions in next state
            online_next_q_preds = self.online_net(next_states)
            # Use eval_net to calculate next_q_preds for actions chosen by online_net
            next_q_preds = self.eval_net(next_states)
        act_q_preds = q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)).squeeze(-1)
        online_actions = online_next_q_preds.argmax(dim=-1, keepdim=True)
        max_next_q_preds = next_q_preds.gather(-1, online_actions).squeeze(-1)
        max_q_targets = batch_rewards + self.gamma * (1 - batch['dones']) * max_next_q_preds
        logger.debug(f'act_q_preds: {act_q_preds}\nmax_q_targets: {max_q_targets}')
        q_loss = self.net.loss_fn(act_q_preds, max_q_targets)

        # TODO use the same loss_fn but do not reduce yet
        if 'Prioritized' in util.get_class_name(self.body.memory):  # PER
            errors = (max_q_targets - act_q_preds.detach()).abs().cpu().numpy()
            self.body.memory.update_priorities(errors)
        return q_loss

    def train(self):
        '''
        Completes one training step for the agent if it is time to train.
        i.e. the environment timestep is greater than the minimum training timestep and a multiple of the training_frequency.
        Each training step consists of sampling n batches from the agent's memory.
        For each of the batches, the target Q values (q_targets) are computed and a single training step is taken k times
        Otherwise this function does nothing.
        '''
        if util.in_eval_lab_modes():
            return np.nan
        clock = self.body.env.clock
        if self.to_train == 1:
            self.reward_agent.eval()
            self.reward_count = 0
            self.batch_count = 0
            total_loss = torch.tensor(0.0)
            for _ in range(self.training_iter):
                batch = self.sample()
                clock.set_batch_size(len(batch))
                for _ in range(self.training_batch_iter):
                    loss = self.calc_q_loss(batch)
                    self.net.train_step(loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
                    total_loss += loss
            loss = total_loss / (self.training_iter * self.training_batch_iter)
            reward_irl = self.reward_count / self.batch_count
            # reset
            print("***********")
            print(reward_irl)
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, frame: {clock.frame}, t: {clock.t}, total_reward so far: {self.body.total_reward}, loss: {loss:g}, irl_reward: {reward_irl}')
            return loss.item()
        else:
            return np.nan

# WDQN this code will be used
class WarmUpDQN(DQN):
    '''
    DQN class

    e.g. algorithm_spec
    "algorithm": {
        "name": "WarmUpDQN",
        "action_pdtype": "Argmax",
        "action_policy": "epsilon_greedy",
        "warmup_epi": 300,
        "explore_var_spec": {
            "name": "linear_decay",
            "start_val": 1.0,
            "end_val": 0.1,
            "start_step": 10,
            "end_step": 1000,
        },
        "gamma": 0.99,
        "training_batch_iter": 8,
        "training_iter": 4,
        "training_frequency": 10,
        "training_start_step": 10
    }
    '''
    def __init__(self, agent, global_nets=None):
        super().__init__(agent, global_nets)
        util.set_attr(self, self.algorithm_spec, [
            'warmup_epi',
        ])
        # create the extra replay memory for warm-up 
        MemoryClass = getattr(memory, self.memory_spec['warmup_name'])
        self.body.warmup_memory = MemoryClass(self.memory_spec, self.body)
        if self.memory_spec['warmup_memory_path']!='':
            import pickle
            self.body.warmup_memory = pickle.load(open(self.memory_spec['warmup_memory_path'], 'rb'))

    @lab_api
    def init_nets(self, global_nets=None):
        super().init_nets(global_nets)

    def warmup_sample(self):
        '''Samples a batch from warm-up memory'''
        batch = self.body.warmup_memory.sample()
        batch = util.to_torch_batch(batch, self.net.device, self.body.warmup_memory.is_episodic)
        return batch
    

    
    def calc_q_loss(self, batch, mask=True):
        '''Compute the Q value loss using predicted and target Q values from the appropriate networks'''
        
        batch_rewards_ori, batch_rewards_log, batch_rewards_log_double, reward_log_minus_one = self.fetch_irl_reward(batch)
        self.reward_count += batch_rewards_ori.mean().item()
        self.batch_count += 1
        # batch_rewards = batch_rewards_ori + batch['rewards']
        batch_rewards = batch_rewards_log + batch['rewards']
        # batch_rewards = batch_rewards_log_double  + batch['rewards']
        # batch_rewards = batch_rewards_ori
        # batch_rewards = batch_rewards_log
        # batch_rewards = batch['rewards']

        states = batch['states']
        next_states = batch['next_states']
        q_preds = self.net(states)
        with torch.no_grad():
            # Use online_net to select actions in next state
            online_next_q_preds = self.online_net(next_states)
            # Use eval_net to calculate next_q_preds for actions chosen by online_net
            next_q_preds = self.eval_net(next_states)
        act_q_preds = q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)).squeeze(-1)
        online_actions = online_next_q_preds.argmax(dim=-1, keepdim=True)
        max_next_q_preds = next_q_preds.gather(-1, online_actions).squeeze(-1)
        max_q_targets = batch_rewards + self.gamma * (1 - batch['dones']) * max_next_q_preds          
        logger.debug(f'act_q_preds: {act_q_preds}\nmax_q_targets: {max_q_targets}')
        q_loss = self.net.loss_fn(act_q_preds, max_q_targets)

        # TODO use the same loss_fn but do not reduce yet
        if 'Prioritized' in util.get_class_name(self.body.memory):  # PER
            errors = (max_q_targets - act_q_preds.detach()).abs().cpu().numpy()
            self.body.memory.update_priorities(errors)
        return q_loss

    def train(self):        
        if util.in_eval_lab_modes():
            return np.nan
        clock = self.body.env.clock
        # import pdb; pdb.set_trace()
        # self.batch_count = 0  
        # print("***********")
        if self.to_train == 1:
            # print("===========")
            self.reward_agent.eval()
            total_loss = torch.tensor(0.0)
            self.reward_count = 0
            self.batch_count = 0  
            for _ in range(self.training_iter):
                batches = []
                warmup = False
                if self.body.warmup_memory.size >= self.body.warmup_memory.batch_size:
                    batches.append(self.warmup_sample())
                    # if self.body.env.clock.frame < 100000:
                    #     batches.append(self.warmup_sample())
                    # else:
                    #     batches.append(self.sample())
                    warmup = True
                if self.body.memory.size >= self.body.memory.batch_size:
                    batches.append(self.sample())
                clock.set_batch_size(sum(len(batch) for batch in batches))
                for idx, batch in enumerate(batches):
                    for _ in range(self.training_batch_iter):
                        loss = self.calc_q_loss(batch, False)                           
                        self.net.train_step(loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
                        total_loss += loss
            loss = total_loss / (self.training_iter * self.training_batch_iter)
            reward_irl = self.reward_count / self.batch_count
            logger.info("***********")
            logger.info(reward_irl)
            # reset
            self.to_train = 0
            logger.info(f'Trained {self.name} at epi: {clock.epi}, warmup_size: {self.body.warmup_memory.size}, memory_size: {self.body.memory.size}, loss: {loss:g}, irl_reward: {reward_irl}')
            # logger.info(f'Trained {self.name} at epi: {clock.epi}, warmup_size: {self.body.warmup_memory.size}, memory_size: {self.body.memory.size}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

# NO USE
class DoubleDQN(DQN):
    '''
    Double-DQN (DDQN) class

    e.g. algorithm_spec
    "algorithm": {
        "name": "DDQN",
        "action_pdtype": "Argmax",
        "action_policy": "epsilon_greedy",
        "explore_var_spec": {
            "name": "linear_decay",
            "start_val": 1.0,
            "end_val": 0.1,
            "start_step": 10,
            "end_step": 1000,
        },
        "gamma": 0.99,
        "training_batch_iter": 8,
        "training_iter": 4,
        "training_frequency": 10,
        "training_start_step": 10
    }
    '''
    @lab_api
    def init_nets(self, global_nets=None):
        super().init_nets(global_nets)
        self.online_net = self.net
        self.eval_net = self.target_net
