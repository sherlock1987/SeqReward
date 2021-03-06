# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from convlab.agent import net
from convlab.agent.algorithm import policy_util
from convlab.agent.algorithm.base import Algorithm
from convlab.agent.net import net_util
from convlab.lib import logger, math_util, util
from convlab.lib.decorator import lab_api
from convlab.agent.algorithm import reward_agent, reward_utils
logger = logger.get_logger(__name__)
import copy
import os
import torch

class Reinforce(Algorithm):
    '''
    Implementation of REINFORCE (Williams, 1992) with baseline for discrete or continuous actions http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
    Adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    Algorithm:
        0. Collect n episodes of data
        1. At each timestep in an episode
            - Calculate the advantage of that timestep
            - Multiply the advantage by the negative of the log probability of the action taken
        2. Sum all the values above.
        3. Calculate the gradient of this value with respect to all of the parameters of the network
        4. Update the network parameters using the gradient

    e.g. algorithm_spec:
    "algorithm": {
        "name": "Reinforce",
        "action_pdtype": "default",
        "action_policy": "default",
        "explore_var_spec": null,
        "gamma": 0.99,
        "entropy_coef_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
        "training_frequency": 1,
    }
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
            policy_loss_coef=1.0,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # theoretically, REINFORCE does not have policy update; but in this implementation we have such option
            'explore_var_spec',
            'gamma',  # the discount factor
            'entropy_coef_spec',
            'policy_loss_coef',
            'training_frequency',
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.explore_var_scheduler = policy_util.VarScheduler(self.explore_var_spec)
        self.body.explore_var = self.explore_var_scheduler.start_val
        if self.entropy_coef_spec is not None:
            self.entropy_coef_scheduler = policy_util.VarScheduler(self.entropy_coef_spec)
            self.body.entropy_coef = self.entropy_coef_scheduler.start_val

    @lab_api
    def init_nets(self, global_nets=None):
        '''
        Initialize the neural network used to learn the policy function from the spec
        Below we automatically select an appropriate net for a discrete or continuous action space if the setting is of the form 'MLPNet'. Otherwise the correct type of network is assumed to be specified in the spec.
        Networks for continuous action spaces have two heads and return two values, the first is a tensor containing the mean of the action policy, the second is a tensor containing the std deviation of the action policy. The distribution is assumed to be a Gaussian (Normal) distribution.
        Networks for discrete action spaces have a single head and return the logits for a categorical probability distribution over the discrete actions
        '''
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


        reward_path = '../irl/NeuralDialog-LAED/logs/2019-08-16T12-04-13-mwoz_gan_vae.py'  # r4
        use_gpu = False
        # self.reward_agent = reward_agent.RewardAgent(use_gpu)
        self.reward_agent = reward_agent.RewardAgent_EncoderSide(use_gpu)
        val_feed = reward_utils.WoZGanDataLoaders('val')
        reward_agent.load_reward_model(self.reward_agent, reward_path, use_gpu)
        if use_gpu:
            self.reward_agent.cuda()
        self.reward_agent.eval()
        self.reward_count = 0
        self.batch_count = 0
        reward_utils.reward_validate(self.reward_agent, val_feed)

    def fetch_irl_reward_batch(self, batch):
        reward = self.reward_agent(copy.deepcopy(batch))
        batch['irl_rewards'] = reward
        return batch
    def fetch_irl_reward(self, batch):
        self.reward_agent.eval()
        with torch.no_grad():        
            reward = self.reward_agent(copy.deepcopy(batch)).detach().view(-1)
        reward_log = torch.log(reward)
        reward_log_double = torch.log(reward) - torch.log(1 - reward)
        return reward, reward_log, reward_log_double
    def modify_batch_reward(self, batch):
        batch_rewards_ori, batch_rewards_log, batch_rewards_log_double = self.fetch_irl_reward(batch)
        self.reward_count += batch_rewards_ori.mean().item()
        self.batch_count += 1
        # batch_rewards = batch_rewards_ori
        # batch_rewards = batch_rewards_ori + batch['rewards']
        batch_rewards = batch_rewards_log + batch['rewards']
        # batch_rewards = batch_rewards_log 
        # batch_rewards = batch_rewards_log_double + batch['rewards']
        
        # batch_rewards = batch['rewards']
        return batch_rewards

    @lab_api
    def calc_pdparam(self, x, net=None):
        '''
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        net = self.net if net is None else net
        pdparam = net(x)
        return pdparam

    @lab_api
    def act(self, state):
        body = self.body
        action = self.action_policy(state, self, body)
        return action.cpu().squeeze().numpy()  # squeeze to handle scalar

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        batch = self.body.memory.sample()
        batch = util.to_torch_batch(batch, self.net.device, self.body.memory.is_episodic)
        return batch

    def calc_pdparam_batch(self, batch):
        '''Efficiently forward to get pdparam and by batch for loss computation'''
        states = batch['states']
        if self.body.env.is_venv:
            states = math_util.venv_unpack(states)
        pdparam = self.calc_pdparam(states)
        return pdparam

    def calc_ret_advs(self, batch):
        '''Calculate plain returns; which is generalized to advantage in ActorCritic'''
        batch_rewards = self.modify_batch_reward(batch)
        rets = math_util.calc_returns(batch_rewards, batch['dones'], self.gamma)
        advs = rets
        if self.body.env.is_venv:
            advs = math_util.venv_unpack(advs)
        logger.debug(f'advs: {advs}')
        return advs

    def calc_policy_loss(self, batch, pdparams, advs):
        '''Calculate the actor's policy loss'''
        action_pd = policy_util.init_action_pd(self.body.ActionPD, pdparams)
        actions = batch['actions']
        if self.body.env.is_venv:
            actions = math_util.venv_unpack(actions)
        log_probs = action_pd.log_prob(actions)
        policy_loss = - self.policy_loss_coef * (log_probs * advs).mean()
        if self.entropy_coef_spec:
            entropy = action_pd.entropy().mean()
            self.body.mean_entropy = entropy  # update logging variable
            policy_loss += (-self.body.entropy_coef * entropy)
        logger.debug(f'Actor policy loss: {policy_loss:g}')
        return policy_loss

    @lab_api
    def train(self):
        if util.in_eval_lab_modes():
            return np.nan
        clock = self.body.env.clock
        if self.to_train == 1:
            batch = self.sample()
            clock.set_batch_size(len(batch))
            pdparams = self.calc_pdparam_batch(batch)
            advs = self.calc_ret_advs(batch)
            loss = self.calc_policy_loss(batch, pdparams, advs)
            self.net.train_step(loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
            # reset
            self.to_train = 0
            logger.info(f'Trained {self.name} at epi: {clock.epi}, frame: {clock.frame}, t: {clock.t}, total_reward so far: {self.body.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    @lab_api
    def update(self):
        self.body.explore_var = self.explore_var_scheduler.update(self, self.body.env.clock)
        if self.entropy_coef_spec is not None:
            self.body.entropy_coef = self.entropy_coef_scheduler.update(self, self.body.env.clock)
        return self.body.explore_var

class WarmUpReinforce(Reinforce):
    '''
    Implementation of REINFORCE (Williams, 1992) with baseline for discrete or continuous actions http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
    Adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    Algorithm:
        0. Collect n episodes of data
        1. At each timestep in an episode
            - Calculate the advantage of that timestep
            - Multiply the advantage by the negative of the log probability of the action taken
        2. Sum all the values above.
        3. Calculate the gradient of this value with respect to all of the parameters of the network
        4. Update the network parameters using the gradient

    e.g. algorithm_spec:
    "algorithm": {
        "name": "Reinforce",
        "action_pdtype": "default",
        "action_policy": "default",
        "warmup_epi": 300,
        "explore_var_spec": null,
        "gamma": 0.99,
        "entropy_coef_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
        "training_frequency": 1,
    }
    '''
    def __init__(self, agent, global_nets=None):
        super().__init__(agent, global_nets)
        util.set_attr(self, self.algorithm_spec, [
            'warmup_epi',
        ])