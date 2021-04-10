from __future__ import print_function
import numpy as np
import logging
import torch
import os
import json
from collections import defaultdict
INT = 0
LONG = 1
FLOAT = 2
class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack


class DataLoader(object):
    logger = logging.getLogger()

    def __init__(self, name, fix_batch=True):
        self.batch_size = 0
        self.ptr = 0
        self.num_batch = None
        self.indexes = None
        self.data_size = None
        self.batch_indexes = None
        self.fix_batch=fix_batch
        self.max_utt_size = None  
        self.name = name

    def _shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, *args, **kwargs):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, config, shuffle=True, verbose=True):
        self.ptr = 0
        self.batch_size = config.batch_size
        self.num_batch = self.data_size // config.batch_size
        if verbose:
            self.logger.info("Number of left over sample %d" % (self.data_size - config.batch_size * self.num_batch))

        # if shuffle and we want to group lines, shuffle batch indexes
        if shuffle and not self.fix_batch:
            self._shuffle_indexes()

        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        if shuffle and self.fix_batch:
            self._shuffle_batch_indexes()

        if verbose:
            self.logger.info("%s begins with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None

    def pad_to(self, max_len, tokens, do_pad=True):
        if len(tokens) >= max_len:
            return tokens[0:max_len - 1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

class WoZGanDataLoaders(DataLoader):
    def __init__(self, name, batch_size=16):
        super(WoZGanDataLoaders, self).__init__(name)
        self.action_num = 300
        self.batch_size=batch_size
        data = self._read_file(name)
        self.data, self.indexes, self.batch_indexes = self.flatten_dialog(data)        
        self.data_size = len(self.data)
        print("Data size: {}".format(self.data_size))
    
    def _read_file(self, dataset):
        # changed part
        # with open(os.path.join('./data/multiwoz', dataset + '.sa.json')) as f:
        # with open(os.path.join('./data/multiwoz', dataset + '.sa_NoHotel.json')) as f: 
        # with open(os.path.join('./data/multiwoz', dataset + '.sa_alldomain.json')) as f:    # ****    
        with open(os.path.join('./data/multiwoz', dataset + '.sa_alldomain_withnext.json')) as f:        
            data = json.load(f)
        return data
     
    def flatten_dialog(self, data):
        results = []
        indexes = []
        batch_indexes = []
        resp_set = set()
        dict_act_seq = defaultdict(list)
        for dlg in data:
            batch_index = []
            state_onehot = dlg['state_onehot']
            state_convlab = dlg['state_convlab']
            action_index = dlg['action']
            action_index_binary = dlg['action_binary']
            state_convlab_next = dlg['state_convlab_next']
            results.append(Pack(action_id=action_index, 
                                state_onehot=state_onehot,
                                action_id_binary=action_index_binary, 
                                state_convlab=state_convlab,
                                state_convlab_next = state_convlab_next
                                ))
            indexes.append(len(indexes))
            batch_index.append(indexes[-1])
            if len(batch_index) > 0:
                batch_indexes.append(batch_index)
        return results, indexes, batch_indexes

    def epoch_init(self, shuffle=True, verbose=True, fix_batch=False):
        self.ptr = 0
        self.num_batch = self.data_size // self.batch_size 
        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size: (i + 1) * self.batch_size])
        if verbose:
            print('Number of left over sample = %d' % (self.data_size - self.batch_size * self.num_batch))
        if shuffle:
            if fix_batch:
                self._shuffle_batch_indexes()
            else:
                self._shuffle_indexes()

        if verbose:
            print('%s begins with %d batches' % (self.name, self.num_batch))

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        keys = []
        metas = []
        index = []
        turn_id = []

        state_onehot, state_convlab, state_convlab_next = [], [], []
        action_id, action_id_binary = [], []
        for row in rows:  
            state_onehot.append(row['state_onehot'])
            state_convlab.append(row['state_convlab'])
            action_id.append(row['action_id'])
            action_id_binary.append(row['action_id_binary'])
            state_convlab_next.append(row['state_convlab_next'])
            
        # state = np.array(state)
        # action = np.array(action)
        return Pack(action_id=action_id, 
                    state_onehot=state_onehot,
                    state_convlab=state_convlab, 
                    action_id_binary=action_id_binary,
                    state_convlab_next = state_convlab_next
                    )


def reward_validate(agent, valid_feed):
    with torch.no_grad():
        agent.eval()
        valid_feed.epoch_init(shuffle=False, verbose=True)
        batch_num = 0
        rew_all = 0
        while True:
            batch = valid_feed.next_batch()
            if batch is None:  
                break                                                                                                                     
            rew = agent.forward_validate(batch)
            # wgan_reward.append(torch.stack(acc))
            rew_all += rew.mean().item()
            batch_num+=1
    # 0.79
    logging.info("Avg reward: {}".format(rew_all/batch_num))
    print("Avg reward: {}".format(rew_all/batch_num))

def kl_divergence(reward_pos, reward_neg):
    """
    Two list
    Returns: one number(float)
    """
    reward_pos = np.array(copy.deepcopy(reward_pos))
    reward_neg = np.array(copy.deepcopy(reward_neg))

    di = reward_pos/reward_neg
    di = np.log(di)
    di *= reward_pos
    kl = np.sum(di)
    return kl
import copy
def js_divergence(reward_pos, reward_neg):
    reward_pos = np.array(copy.deepcopy(reward_pos))
    reward_neg = np.array(copy.deepcopy(reward_neg))
    return (kl_divergence(reward_pos, reward_neg)+kl_divergence(reward_neg, reward_pos))/2

def plot_graph(agent, valid_feed, surgery="hard", name="mine"):
    """
    Args:
        agent: mine_2/mine_3, ziming.
        valid_feed: test data
        surgery: method of computation.
        name:ziming or mine.
    Returns:
    """
    states, machine_act, human_act = build_fake_data(agent, valid_feed)
    embed_rep = agent.vae.get_embed(states)
    with torch.no_grad():
        reward_neg = agent.discriminator(embed_rep, machine_act, surgery).tolist()
        reward_pos = agent.discriminator(embed_rep, human_act, surgery).tolist()
    import seaborn as sns
    import numpy as np
    from numpy.random import randn
    import matplotlib.pyplot as plot
    from matplotlib.backends.backend_pdf import PdfPages

    fig = plot.figure(figsize=(20,15))
    ax = fig.add_subplot(1, 1, 1)

    """
    Do hyper paramater over here.
    """
    bins_num = 300
    upper_bound = 0.2
    font_size = 44

    reward_neg = np.array(reward_neg)*bins_num
    reward_pos = np.array(reward_pos)*bins_num
    sns.distplot(reward_neg, bins=int(bins_num), kde=False, rug=False, hist=True, norm_hist= True, color = "dodgerblue", label = "NEG")
    sns.distplot(reward_pos, bins=int(bins_num), kde=False, rug=False, hist=True, norm_hist= True, color = "lightcoral", label = "POS")

    # ax.set_xticks([0, 25, 50, 75, 100])
    ticks = np.array([0, 25, 50, 75, 100])*(bins_num//100)
    ax.set_xticks(ticks)
    ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
    plot.ylim(0, upper_bound)
    def forward(x):
        return x ** (0.6)

    def inverse(x):
        return x ** (1/0.6)

    ax.set_yscale('function', functions=(forward, inverse))

    # Set Size.
    plot.xticks(fontsize=font_size)
    plot.yticks(fontsize=font_size)
    # ax.set_xlabel("Score", font_size=font_size)
    # ax.set_ylabel("Probability", font_size=font_size)
    plot.xlabel("Score", fontsize = font_size)
    plot.ylabel("Probability", fontsize =font_size)
    plot.legend(fontsize=font_size)

    if name == "ziming":
        new_name = "ziming"
    else:
        new_name = name + "_" + surgery
    with PdfPages('{}.pdf'.format(new_name)) as pp:
        pp.savefig(plot.gcf())
    plot.show()

def plot_graph_4(agent_mine, agent_ziming, valid_feed, type_list):
    """
    Args:
        agent: mine and ziming's agent
        valid_feed:
        type_list:
    Returns:
    """
    import seaborn as sns
    import numpy as np
    from numpy.random import randn
    import matplotlib as mpl
    import matplotlib.pyplot as plot
    from matplotlib.backends.backend_pdf import PdfPages


    # get data, and do embedding.
    states, machine_act, human_act = build_fake_data(agent_mine, valid_feed)
    embed_rep = agent_mine.vae.get_embed(states)
    with torch.no_grad():
        reward_neg_1 = agent_mine.discriminator(embed_rep, machine_act, type_list[0]).tolist()
        reward_pos_1 = agent_mine.discriminator(embed_rep, human_act, type_list[0]).tolist()
        reward_neg_2 = agent_mine.discriminator(embed_rep, machine_act, type_list[1]).tolist()
        reward_pos_2 = agent_mine.discriminator(embed_rep, human_act, type_list[1]).tolist()
        reward_neg_3 = agent_mine.discriminator(embed_rep, machine_act, type_list[2]).tolist()
        reward_pos_3 = agent_mine.discriminator(embed_rep, human_act, type_list[2]).tolist()

    print(type_list[0], np.mean(reward_neg_1), np.mean(reward_pos_1))
    print(type_list[1], np.mean(reward_neg_2), np.mean(reward_pos_2))
    print(type_list[2], np.mean(reward_neg_3), np.mean(reward_pos_3))

    # plot.figure(figsize=(10, 10))
    fig, axes = plot.subplots(2, 2)
    # set the size of figure
    plot.subplots_adjust(wspace=0.5, hspace=0.5)

    sns.distplot(reward_neg_1, bins=int(100), kde=False, rug=False, hist=True, color = "dodgerblue", label = "NEG", ax =axes[0][0])
    sns.distplot(reward_pos_1, bins=int(100), kde=False, rug=False, hist=True, color = "lightcoral", label = "POS", ax =axes[0][0])

    # hist = False, kde = True, rug = True
    sns.distplot(reward_neg_2, bins=int(100), kde=False, rug=False, hist=True, color = "dodgerblue", label = "NEG", ax =axes[0][1])
    sns.distplot(reward_pos_2, bins=int(100), kde=False, rug=False, hist=True, color = "lightcoral", label = "POS", ax =axes[0][1])

    sns.distplot(reward_neg_3, bins=int(100), kde=False, rug=False, hist=True, color = "dodgerblue", label = "NEG", ax =axes[1][0])
    sns.distplot(reward_pos_3, bins=int(100), kde=False, rug=False, hist=True, color = "lightcoral", label = "POS", ax =axes[1][0])

    embed_rep = agent_ziming.vae.get_embed(states)
    with torch.no_grad():
        reward_neg_4 = agent_ziming.discriminator(embed_rep, machine_act).tolist()
        reward_pos_4 = agent_ziming.discriminator(embed_rep, human_act).tolist()

    print(np.mean(reward_neg_4), np.mean(reward_pos_4))


    sns.distplot(reward_neg_4, bins=int(100), kde=False, rug=False, hist=True, color="dodgerblue", label="NEG",
                 ax=axes[1][1])
    sns.distplot(reward_pos_4, bins=int(100), kde=False, rug=False, hist=True, color="lightcoral", label="POS",
                 ax=axes[1][1])

    # set label stuff.
    axes[0,0].set_xlabel("Probability")
    axes[0,0].set_ylabel("Frequency")
    axes[0,1].set_xlabel("Probability")
    axes[0,1].set_ylabel("Frequency")
    axes[1,0].set_xlabel("Probability")
    axes[1,0].set_ylabel("Frequency")
    axes[1,1].set_xlabel("Probability")
    axes[1,1].set_ylabel("Frequency")


    # plot.legend()


    with PdfPages('tsne.pdf') as pp:
        pp.savefig(plot.gcf())
    plot.show()
    pass

def plot_graph_4_seperate(agent_mine, agent_ziming, valid_feed, type_list):
    """
    Args:
        agent: mine and ziming's agent
        valid_feed:
        type_list:
    Returns:
    """
    import seaborn as sns
    import numpy as np
    from numpy.random import randn
    import matplotlib as mpl
    import matplotlib.pyplot as plot
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_recall_curve
    from matplotlib.ticker import NullFormatter, FixedLocator
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    # hyper paramaters
    bins_num = 200


    # get data, and do embedding.
    states, machine_act, human_act = build_fake_data(agent_mine, valid_feed)
    embed_rep = agent_mine.vae.get_embed(states)
    with torch.no_grad():
        reward_neg_1 = agent_mine.discriminator(embed_rep, machine_act, type_list[0]).view(-1).tolist()
        reward_pos_1 = agent_mine.discriminator(embed_rep, human_act, type_list[0]).view(-1).tolist()
        reward_neg_2 = agent_mine.discriminator(embed_rep, machine_act, type_list[1]).view(-1).tolist()
        reward_pos_2 = agent_mine.discriminator(embed_rep, human_act, type_list[1]).view(-1).tolist()
        reward_neg_3 = agent_mine.discriminator(embed_rep, machine_act, type_list[2]).view(-1).tolist()
        reward_pos_3 = agent_mine.discriminator(embed_rep, human_act, type_list[2]).view(-1).tolist()

        # reward_pos_1 = np.log(reward_pos_1).tolist()
        # reward_neg_1 = np.log(reward_neg_1).tolist()
        # reward_pos_1 = (np.log(reward_pos_1)-np.log(1.0-np.array(reward_pos_1))).tolist()
        # reward_neg_1 = (np.log(reward_neg_1)-np.log(1.0-np.array(reward_neg_1))).tolist()
        
    embed_rep = agent_ziming.vae.get_embed(states)
    with torch.no_grad():
        reward_neg_4 = agent_ziming.discriminator(embed_rep, machine_act).view(-1).tolist()
        reward_pos_4 = agent_ziming.discriminator(embed_rep,   human_act).view(-1).tolist()

        # reward_pos_4 = (np.log(reward_pos_4)-np.log(1.0-np.array(reward_pos_4))).tolist()
        # reward_neg_4 = (np.log(reward_neg_4)-np.log(1.0-np.array(reward_neg_4))).tolist()

    fig, axes = plot.subplots(2, 2)
    # set the size of figure
    plot.subplots_adjust(wspace=0.5, hspace=0.5)
    """
    plot
    """
    # # print mean
    # print(type_list[0], np.mean(reward_neg_1), np.mean(reward_pos_1))
    # print(type_list[1], np.mean(reward_neg_2), np.mean(reward_pos_2))
    # print(type_list[2], np.mean(reward_neg_3), np.mean(reward_pos_3))
    # print(np.mean(reward_neg_4), np.mean(reward_pos_4))
    """
    calculate the number over here, three + JS
    """
    label_y = [0] * len(reward_neg_1)+[1]*len(reward_pos_1)
    y_label= [(label_y[i]) > 0.5 for i in range(len(label_y))]

    reward = copy.deepcopy(reward_neg_1) + copy.deepcopy(reward_pos_1)
    y_pred = [(reward[i]) > 0.5 for i in range(len(reward))]
    prec, rec, f1, _ = precision_recall_fscore_support(y_pred, y_label, average="binary")
    print(type_list[0]+": {:.3f} & {:.3f} & {:.3f} & {:.3f}".format(prec, rec, f1, js_divergence(reward_pos_1, reward_neg_1)/len(reward_neg_1)))
    acc = accuracy_score(label_y, y_pred)
    # print(type_list[0]+": {:.3f} & {:.3f} & {:.3f}".format(prec, rec, f1))
    print(accuracy_score(label_y, y_pred))
    print(confusion_matrix(label_y, y_pred))

    reward = copy.deepcopy(reward_neg_2) + copy.deepcopy(reward_pos_2)
    y_pred = [(reward[i]) > 0.5 for i in range(len(reward))]
    prec, rec, f1, _ = precision_recall_fscore_support(y_pred, y_label, average="binary")
    print(type_list[1]+": {:.3f} & {:.3f} & {:.3f} & {:.3f}".format(prec, rec, f1, js_divergence(reward_pos_2, reward_neg_2)/len(reward_neg_1)))
    # print(type_list[1]+": {:.3f} & {:.3f} & {:.3f}".format(prec, rec, f1))
    print(accuracy_score(label_y, y_pred))
    print(confusion_matrix(label_y, y_pred))

    reward = copy.deepcopy(reward_neg_3) + copy.deepcopy(reward_pos_3)
    y_pred = [(reward[i]) > 0.5 for i in range(len(reward))]
    prec, rec, f1, _ = precision_recall_fscore_support(y_pred, y_label, average="binary")
    print(type_list[2]+": {:.3f} & {:.3f} & {:.3f} & {:.3f}".format(prec, rec, f1, js_divergence(reward_pos_3, reward_neg_3)/len(reward_neg_1)))
    print(accuracy_score(label_y, y_pred))
    print(confusion_matrix(label_y, y_pred))

    reward = copy.deepcopy(reward_neg_4) + copy.deepcopy(reward_pos_4)
    y_pred = [(reward[i]) > 0.5 for i in range(len(reward))]
    prec, rec, f1, _ = precision_recall_fscore_support(y_pred, y_label, average="binary")
    print("ziming: {:.3f} & {:.3f} & {:.3f} & {:.3f} ".format(prec, rec, f1, js_divergence(reward_pos_4, reward_neg_4)/len(reward_neg_1)))
    # print("ziming: {:.3f} & {:.3f} & {:.3f}".format(prec, rec, f1))
    print(accuracy_score(label_y, y_pred))
    print(confusion_matrix(label_y, y_pred))


    sns.distplot(reward_neg_1, bins=int(bins_num), kde=False, rug=False, hist=True, color = "dodgerblue", label = "NEG", ax =axes[0][0])
    sns.distplot(reward_pos_1, bins=int(bins_num), kde=False, rug=False, hist=True, color = "lightcoral", label = "POS", ax =axes[0][0])
    plot.ylim(0, 2500)


    sns.distplot(reward_neg_2, bins=int(bins_num), kde=False, rug=False, hist=True, color = "dodgerblue", label = "NEG", ax =axes[0][1])
    sns.distplot(reward_pos_2, bins=int(bins_num), kde=False, rug=False, hist=True, color = "lightcoral", label = "POS", ax =axes[0][1])
    plot.ylim(0, 2500)

    sns.distplot(reward_neg_3, bins=int(bins_num), kde=False, rug=False, hist=True, color = "dodgerblue", label = "NEG", ax =axes[1][0])
    sns.distplot(reward_pos_3, bins=int(bins_num), kde=False, rug=False, hist=True, color = "lightcoral", label = "POS", ax =axes[1][0])
    plot.ylim(0, 2500)

    sns.distplot(reward_neg_4, bins=int(bins_num), kde=False, rug=False, hist=True, color="dodgerblue", label="NEG",
                 ax=axes[1][1])
    sns.distplot(reward_pos_4, bins=int(bins_num), kde=False, rug=False, hist=True, color="lightcoral", label="POS",
                 ax=axes[1][1])

    plot.ylim(0, 2500)

    def forward(x):
        return x ** (1/2)

    def inverse(x):
        return x ** (2)

    axes[0,0].set_yscale('function', functions=(forward, inverse))
    axes[0,1].set_yscale('function', functions=(forward, inverse))
    axes[1,0].set_yscale('function', functions=(forward, inverse))
    axes[1,1].set_yscale('function', functions=(forward, inverse))

    # set label stuff.
    axes[0,0].set_xlabel("score")
    axes[0,0].set_ylabel("Frequency")
    axes[0,1].set_xlabel("score")
    axes[0,1].set_ylabel("Frequency")
    axes[1,0].set_xlabel("score")
    axes[1,0].set_ylabel("Frequency")
    axes[1,1].set_xlabel("score")
    axes[1,1].set_ylabel("Frequency")

    axes[0,1].yaxis.set_major_locator(FixedLocator(np.arange(0, 50, 10)**2))
    axes[1,0].yaxis.set_major_locator(FixedLocator(np.arange(0, 50, 10)**2))
    axes[1,1].yaxis.set_major_locator(FixedLocator(np.arange(0, 50, 10)**2))

    # plot.legend()
    with PdfPages('tsne.pdf') as pp:
        pp.savefig(plot.gcf())

    plot.show()
    plot.clf()

    pass


def get_point_list(agent, valid_feed, random = False):
    record = []
    with torch.no_grad():
        agent.eval()
        valid_feed.epoch_init(shuffle=True, verbose=False)
        batch_num = 0
        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break
            rew = agent.forward_validate(batch)
            # wgan_reward.append(torch.stack(acc))
            record+=rew.tolist()
            batch_num+=1
    return record

def one_hot_embedding(labels, num_classes):
    # print(labels)
    if type(labels)==list:
        labels = torch.LongTensor(labels)
    y = torch.eye(num_classes)
    return y[labels]

def build_fake_data(agent, valid_feed):
    """
    This function will return machine: human = [state, action_randperm] : [state, action_noshuffle]
    Args:
        agent:
        valid_feed:
    Returns: states, machine_act, human_act
    """
    sample_shape = 64, 128, 128
    import pickle
    import random
    with open("/home/raleigh/work/Dp-without-Adv/convlab_repo/convlab/agent/algorithm/random_rule.pkl", 'rb') as f:
        random_rule = pickle.load(f)
    with torch.no_grad():
        agent.eval()
        valid_feed.epoch_init(shuffle=True, verbose=False)
        states = torch.tensor([])
        machine_act = torch.tensor([])
        human_act = torch.tensor([])
        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break
            real_state_rep = batch["state_convlab"]
            machine_state_rep = real_state_rep
            action_id_list = batch["action_id"]
            action_data_feed = one_hot_embedding(batch["action_id"], 300)
            # get random act, super random.
            idx_random_list = []
            for i, id in enumerate(action_id_list):
                length = len(random_rule[id])
                idx_random = random.randint(0, length)
                idx_random_list.append(idx_random)

            machine_action_rep = one_hot_embedding(idx_random_list, 300)
            # machine_action_rep = action_data_feed[torch.randperm(action_data_feed.size()[0]), :]

            state_rep = torch.tensor(real_state_rep)
            states = torch.cat((states, state_rep), dim = 0)
            machine_act = torch.cat((machine_act, machine_action_rep), dim = 0)
            human_act = torch.cat((human_act, action_data_feed), dim = 0)

        return states, machine_act, human_act


def one_hot_embedding(labels, num_classes):
    # print(labels)
    if type(labels)==list:
        labels = torch.LongTensor(labels)
    y = torch.eye(num_classes) 
    return y[labels] 