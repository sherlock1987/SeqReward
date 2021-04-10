import argparse
import os
import numpy as np
import math
import sys
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
# from laed import nn_lib
# from laed.enc2dec import decoders
# from laed.enc2dec.decoders import DecoderRNN
from laed.enc2dec.encoders import EncoderRNN, RnnUttEncoder
from laed.dataset.corpora import PAD, EOS, EOT, BOS
from laed.utils import INT, FLOAT, LONG, cast_type, Pack
from gan.torch_utils import GumbelConnector, LayerNorm
"""
Contains VAE model and also the basic G and D.
"""


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.use_gpu = config.use_gpu
        self.flush_valid = False
        self.config = config
        self.kl_w = 0.0
        self.gumbel_connector=GumbelConnector(config.use_gpu)

    def cast_gpu(self, var):
        if self.use_gpu:
            return var.cuda()
        else:
            return var.cpu()

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        if type(inputs)==list:
            return cast_type(Variable(torch.Tensor(inputs)), dtype, self.use_gpu)
        return cast_type(Variable(torch.from_numpy(inputs)), dtype, self.use_gpu)


    def forward(self, *input):
        raise NotImplementedError

    def backward(self, batch_cnt, loss):
        """
        Args:
            batch_cnt: no use
            loss: {vae_loss: 720. other_losses}

        Returns:

        """
        total_loss = self.valid_loss(loss, batch_cnt)
        total_loss += self.l2_norm()
        total_loss.backward()
        self.clip_gradient()

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

class Generator(BaseModel):
    def __init__(self, config):
        super(Generator, self).__init__(config)
        
        state_in_size = config.state_noise_dim
        action_in_size = config.action_noise_dim
        state_action_out_size = config.vae_embed_size

        self.state_action_model = nn.Sequential(
            # original: 5 block + 1 linear
            # nn.Dropout(config.dropout),  
            
            nn.Linear(state_in_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Dropout(config.dropout),  
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            # nn.Dropout(config.dropout),  
            
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            # nn.Dropout(config.dropout),  
            
            # nn.Linear(64, 64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(True),
            # nn.Dropout(config.dropout),  


            nn.Linear(64, state_action_out_size),
            nn.Tanh()
            )
    def backward(self, batch_cnt, loss):
        total_loss = self.valid_loss(loss, batch_cnt)
        total_loss.backward()
        # self.clip_gradient()
        
    def forward(self, s_z, a_z):
        state_action_embedding = self.state_action_model(self.cast_gpu(s_z))
        return state_action_embedding

class Discriminator(BaseModel):
    def __init__(self, config):
        super(Discriminator, self).__init__(config)
        config.dropout = 0.3
        self.state_in_size = config.state_out_size
        self.action_in_size = 9
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size/2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size/2)
        
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config.dropout),

            nn.Linear(self.state_in_size/2 + self.action_in_size/2, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),

            nn.Linear(64, 1)
        )

    def decay_noise(self):
        self.noise_input *= 0.995

    def forward(self, state, action):
        state_1 = self.state_rep(self.cast_gpu(state))
        action_1 = self.action_rep(self.cast_gpu(action))
        # print(state.shape, action_1.shape)
        state_action = torch.cat([state_1, action_1], 1)
        validity = torch.sigmoid(self.model(state_action))
        validity = torch.clamp(validity, 1e-8, 1-1e-8)
        return validity

class Discriminator_SA(BaseModel):
    def __init__(self, config):
        super(Discriminator_SA, self).__init__(config)
        config.dropout = 0.3
        self.input_size = config.vae_embed_size
        self.noise_input = 0.01
        self.model = nn.Sequential(
            
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            

            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(config.dropout),

            nn.Linear(32, 1)
        )
    def decay_noise(self):
        self.noise_input *= 0.995

    def forward(self, state_action):
        validity = torch.sigmoid(self.model(self.cast_gpu(state_action)))
        validity = torch.clamp(validity, 1e-8, 1-1e-8)
        return validity
    
class Discriminator_StateAction(BaseModel):
    def __init__(self, config):
        super(Discriminator_StateAction, self).__init__(config)
        config.dropout = 0.3
        self.input_size = config.vae_embed_size
        self.noise_input = 0.01
        self.model = nn.Sequential(
         
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(config.dropout),

            nn.Linear(32, 1)
        )

    def decay_noise(self):
        self.noise_input *= 0.995

    def forward(self, state_action):
        validity = torch.sigmoid(self.model(self.cast_gpu(state_action)))
        validity = torch.clamp(validity, 1e-8, 1-1e-8)
        return validity

class ContEncoder(BaseModel):
    def __init__(self, corpus, config):
        super(ContEncoder, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.config = config

        self.embedding = nn.Embedding(self.vocab_size, config.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.utt_encoder = RnnUttEncoder(config.utt_cell_size, 0.0,
                                        bidirection=False,
                                         #  bidirection=True in the original code
                                         use_attn=config.utt_type == 'attn_rnn',
                                         vocab_size=self.vocab_size,
                                         embedding=self.embedding)

        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size,
                                      config.ctx_cell_size,
                                      0.0,
                                      0.0,
                                      config.num_layer,
                                      config.rnn_cell,
                                      variable_lengths=self.config.fix_batch)


    def forward(self, data_feed):
        ctx_lens = data_feed['context_lens']
        batch_size = len(ctx_lens)

        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        
        # context encoder
        if self.config.hier:
            c_inputs = self.utt_encoder(ctx_utts)
            c_outs, c_last = self.ctx_encoder(c_inputs, ctx_lens)
            c_last = c_last.squeeze(0)
        else:
            c_inputs = self.utt_encoder(ctx_utts)
            c_last = c_inputs.squeeze(1)
        return c_last

# this class is to convert the original action to the corresponding latent action
class ActionEncoder(BaseModel):
    def __init__(self, corpus, config, name2action):
        super(ActionEncoder, self).__init__(config)
        self.name2action_dict = name2action
        self.action_num = config.action_num
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        # if self.action_num != len(self.name2action_dict.keys()) + 1:
        #     raise ValueError("the action space should include one spare action to cover some actions \
        #                       that are not in any learned action clusters ")

        self.x_embedding = nn.Embedding(self.vocab_size, config.embed_size)
        self.x_encoder = EncoderRNN(config.embed_size, config.dec_cell_size,
                                    dropout_p=config.dropout,
                                    rnn_cell=config.rnn_cell,
                                    variable_lengths=False)
        self.q_y = nn.Linear(config.dec_cell_size, config.y_size * config.k)
        self.config =config

    def qzx_forward(self, out_utts):
        # this func will be used to extract latent action z given original actions x later in whole pipeline
        output_embedding = self.x_embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        x_last = x_last.transpose(0, 1).contiguous().view(-1, self.config.dec_cell_size)
        qy_logits = self.q_y(x_last).view(-1, self.config.k)
        log_qy = F.log_softmax(qy_logits, dim=1)
        return Pack(qy_logits=qy_logits, log_qy=log_qy)
    
    def forward(self, out_utts):
        out_utts = self.np2var(out_utts, LONG)
        results = self.qzx_forward(self.cast_gpu(out_utts))
        log_qy = results.log_qy.view(-1, self.config.y_size, self.config.k)
        qy = torch.exp(log_qy)
        qy = qy.cpu().data.numpy()
        
        action_list = []
        action_name_list = []
        for b_id in range(out_utts.shape[0]):
            code = []
            for y_id in range(self.config.y_size):
                for k_id in range(self.config.k):
                    if qy[b_id, y_id, k_id] == np.max(qy[b_id, y_id]):
                        code.append(str(k_id))
                        break
            code = '-'.join(code)
            action_id = self.Lookup_action_index(code)
            action_name_list.append(code)
            action_list.append(action_id)
        return action_list, action_name_list

    def Lookup_action_index(self, code):
        if code in self.name2action_dict.keys():
            return self.name2action_dict[code]
        else:
            return self.action_num-1
        
class VAE(BaseModel):
    def __init__(self, config):
        super(VAE, self).__init__(config)
        self.config = config
        self.vae_in_size = config.state_out_size
        self.vae_embed_size = config.vae_embed_size
        
        self.encode_model = nn.Sequential(
            nn.Linear(self.vae_in_size, self.vae_in_size/4),
            nn.ReLU(True),    
        )
        self.decode_model = nn.Sequential(
            nn.Linear(self.vae_embed_size, self.vae_in_size/4),
            nn.ReLU(True),
            nn.Linear(self.vae_in_size/4, self.vae_in_size),
        )
        
        
        self.fc21 = nn.Linear(self.vae_in_size/4, self.vae_embed_size)
        self.fc22 = nn.Linear(self.vae_in_size/4, self.vae_embed_size)

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

class VAE_StateActionEmbed(VAE):
    def __init__(self, config):
        super(VAE_StateActionEmbed, self).__init__(config)
        self.config = config
        self.vae_in_size = config.state_out_size + 100
        self.vae_embed_size = config.vae_embed_size
        
        self.state_model_encode = nn.Sequential(
            nn.Linear(config.state_out_size, config.state_out_size//2),
            nn.ReLU(True)
        )
        self.action_model_encode = nn.Sequential(
            nn.Linear(100, 100//2),
            nn.ReLU(True)
        )

        self.encode_model = nn.Sequential(
            nn.Linear(self.vae_in_size//2, self.vae_in_size//4),
            nn.ReLU(True),
   
        )
        self.decode_model = nn.Sequential(
            nn.Linear(self.vae_embed_size, self.vae_in_size//4),
            nn.ReLU(True),
            nn.Linear(self.vae_in_size//4, self.vae_in_size//2),
            nn.ReLU(True),
        )

        self.state_model_decode = nn.Sequential(
            nn.Linear(self.vae_in_size//2, config.state_out_size),
        )
        self.action_model_decode = nn.Sequential(
            nn.Linear(self.vae_in_size//2, 100)
        )

        
        
        self.fc21 = nn.Linear(self.vae_in_size/4, self.vae_embed_size)
        self.fc22 = nn.Linear(self.vae_in_size/4, self.vae_embed_size)


    def encode(self, x):
        x = x.split(392, 1)
        s = self.state_model_encode(x[0])
        a = self.action_model_encode(x[1])
        sa = torch.cat([s, a], -1)
        h = self.encode_model(sa)
        return self.fc21(h), self.fc22(h)

    def decode(self, z):
        h = self.decode_model(z)
        s = torch.sigmoid(self.state_model_decode(h))
        a = torch.sigmoid(self.action_model_decode(h))
        return torch.cat([s, a], -1)

class VAE_StateActionEmbedMerged(VAE):
    def __init__(self, config):
        super(VAE_StateActionEmbedMerged, self).__init__(config)
        self.config = config
        self.vae_in_size = config.state_out_size + 100
        self.vae_embed_size = config.vae_embed_size

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

    def encode(self, x):
        h = self.encode_model(x)
        return self.fc21(h), self.fc22(h)

    def decode(self, z):
        h = self.decode_model(z)
        return torch.sigmoid(h)

class AutoEncoder(BaseModel):
    def __init__(self, config):
        super(AutoEncoder, self).__init__(config)
        self.config = config
        # self.vae_in_size = config.state_out_size + 9
        self.vae_in_size = config.state_out_size
        self.vae_embed_size = config.vae_embed_size
        
        self.encode_model = nn.Sequential(
            nn.Dropout(config.dropout),          
            nn.Linear(self.vae_in_size, self.vae_in_size/2),
            nn.Tanh(),
            nn.Dropout(config.dropout),                      
            nn.Linear(self.vae_in_size/2, self.vae_embed_size),
            nn.Tanh(), 
        )
        self.decode_model = nn.Sequential(
            nn.Dropout(config.dropout),                  
            nn.Linear(self.vae_embed_size, self.vae_in_size/2),
            nn.Sigmoid(),
            nn.Dropout(config.dropout),                      
            nn.Linear(self.vae_in_size/2, self.vae_in_size),
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
        self.gumbel_connector = GumbelConnector(config.use_gpu)


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
        action_pred = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)
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

class VAE_3(BaseModel):
    def __init__(self, config):
        super(VAE_3, self).__init__(config)
        self.config = config
        # self.vae_in_size = config.state_out_size + 9
        self.vae_in_size = config.state_out_size
        self.vae_embed_size = config.vae_embed_size
        self.embedding_size = config.vae_embed_size

        self.common_model = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.vae_in_size, 256),
            nn.Tanh())

        """
        # (Embedding)_part
        """
        self.e_1 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, self.embedding_size),
            nn.LeakyReLU())

        self.e_2 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, self.embedding_size),
            nn.LeakyReLU())

        self.e_3 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, self.embedding_size),
            nn.LeakyReLU())

        """
        VAE part
        """
        self.hidden2mean_1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.hidden2logv_1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.latent2hidden_1 = nn.Linear(self.embedding_size, self.embedding_size)

        self.hidden2mean_2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.hidden2logv_2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.latent2hidden_2 = nn.Linear(self.embedding_size, self.embedding_size)

        self.hidden2mean_3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.hidden2logv_3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.latent2hidden_3 = nn.Linear(self.embedding_size, self.embedding_size)

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
        self.decode = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * 3, 256),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 392))
        # output Layer
        self.softmax = nn.Softmax(dim=-1)
        self.gumbel_connector = GumbelConnector(config.use_gpu)

    def forward(self, state, action):
        """
        Args:
            state: belief states
            action: action space
        Returns: Decoding and Encoding bf states.
        """
        batch_size = state.size(0)
        # common_h = self.e_common(state)
        # [B, 64] -> [B, seq_len, 64] = [B, 3, 64]
        state_h = self.common_model(state)
        d_e = self.e_1(state_h)
        a_e = self.e_2(state_h)
        s_e = self.e_3(state_h)
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
        domain_pred = self.softmax(self.c_1(mean_1))
        action_pred = self.softmax(self.c_2(mean_2))
        slot_pred  =  self.softmax(self.c_3(mean_3))

        d_vae_e = self.latent2hidden_1(z_1)
        a_vae_e = self.latent2hidden_2(z_2)
        s_vae_e = self.latent2hidden_3(z_3)

        e_vae = torch.cat((d_vae_e, a_vae_e, s_vae_e), dim = -1)
        action_h = self.c_action(e_vae)
        action_pred_whole = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)
        recon_batch = self.decoder(e_vae)
        # return all of this stuff.
        return recon_batch, domain_pred, action_pred, slot_pred, action_pred_whole

    def encoder(self, state, action):
        """
        Args:
            state: belief states
            action: action space
        Returns: Four parts, (domain_VAE) (action_VAE) (slot_VAE) (prediction_[d: a: s])
        """
        batch_size = state.size(0)
        # common_h = self.e_common(state)
        # [B, 64] -> [B, seq_len, 64] = [B, 3, 64]
        state_h = self.common_model(state)
        d_e = self.e_1(state_h)
        a_e = self.e_2(state_h)
        s_e = self.e_3(state_h)
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
        logvar_1 = torch.exp(0.5 * logv_1)
        z_1 = z_1 * logvar_1 + mean_1

        mean_2 = self.hidden2mean_2(a_e)
        logv_2 = self.hidden2logv_2(a_e)
        z_2 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        logvar_2 = torch.exp(0.5 * logv_2)
        z_2 = z_2 * logvar_2 + mean_2

        mean_3 = self.hidden2mean_3(s_e)
        logv_3 = self.hidden2logv_3(s_e)
        z_3 = self.cast_gpu(torch.randn([batch_size, self.embedding_size]))
        logvar_3 = torch.exp(0.5 * logv_3)
        z_3 = z_3 * logvar_3 + mean_3

        # classification should over here.
        domain_pred = self.softmax(self.c_1(mean_1))
        action_pred = self.softmax(self.c_2(mean_2))
        slot_pred  =  self.softmax(self.c_3(mean_3))

        return (mean_1, logvar_1, z_1), (mean_2, logvar_2, z_2), (mean_3, logvar_3, z_3), (domain_pred, action_pred, slot_pred)


    def decoder(self, d_z, a_z, s_z):
        d_vae_e = self.latent2hidden_1(d_z)
        a_vae_e = self.latent2hidden_2(a_z)
        s_vae_e = self.latent2hidden_3(s_z)

        # Todo, add the classification if you need that one.
        e_vae = torch.cat((d_vae_e, a_vae_e, s_vae_e), dim = -1)
        # action_h = self.c_action(e_vae)
        # action_pred_whole = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)
        recon_batch = self.decode(e_vae)
        # return all of this stuff.
        return recon_batch

    def get_embed(self, state):
        """
        Args:
            state:
        Returns:
        """
        batch_size = state.size(0)
        # common_h = self.e_common(state)
        # [B, 64] -> [B, seq_len, 64] = [B, 3, 64]
        state_h = self.common_model(state)
        d_e = self.e_1(state_h)
        a_e = self.e_2(state_h)
        s_e = self.e_3(state_h)
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
        output = torch.cat((mean_1, mean_2, mean_3), dim = -1)
        return output



class AE_3_seq(BaseModel):
    def __init__(self, config):
        super(AE_3_seq, self).__init__(config)
        self.config = config
        self.vae_in_size = config.state_out_size
        self.input_size = config.state_out_size
        self.embedding_size = config.vae_embed_size
        self.temperature = config.gumbel_temp

        self.e_common = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, self.embedding_size),
            nn.LeakyReLU())
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
        self.gumbel_connector = GumbelConnector(config.use_gpu)


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
        action_pred = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)
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

class AE_3_parrallel(BaseModel):
    def __init__(self, config):
        super(AE_3_parrallel, self).__init__(config)
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

        # (Embedding)_part
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

        # Encoder Part
        if config.vae_encoder == 'rnn':
            rnn = nn.RNN
        elif config.vae_encoder == 'gru':
            rnn = nn.GRU
        self.bidirectional = True
        self.num_layers = 3
        self.encoder_rnn = rnn(self.embedding_size, self.embedding_size, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)
        # self.hidden_factor = (2 if self.bidirectional else 1) * self.num_layers
        self.hidden_factor_whole = (2 if self.bidirectional else 1)
        self.hidden_factor_hidden =  (2 if self.bidirectional else 1) * self.num_layers
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

        self.c_1_last = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * self.hidden_factor_whole, self.embedding_size),

            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 17))

        self.c_2_last = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * self.hidden_factor_whole, self.embedding_size),

            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 34))

        self.c_3_last = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * self.hidden_factor_whole, self.embedding_size),

            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 82))


        # Todo: try which one is better, use three to predict one or use last hidden to predict actions?
        # self.c_whole = nn.Sequential(
        #     nn.Dropout(config.dropout),
        #     nn.Linear(self.embedding_size * self.hidden_factor * 3, self.embedding_size * self.hidden_factor *3),
        #
        #     nn.LeakyReLU(),
        #     nn.Dropout(config.dropout),
        #     nn.Linear(self.embedding_size * self.hidden_factor * 3, 300))

        # 128 -> 300
        self.c_tiny = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * self.hidden_factor_whole, self.embedding_size * self.hidden_factor_whole * 4),

            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * self.hidden_factor_whole * 4, 300))

        self.decoder = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * self.hidden_factor_whole *3, self.embedding_size * self.hidden_factor_whole *3),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size * self.hidden_factor_whole *3, self.input_size),
            nn.LeakyReLU())

        # output Layer
        self.softmax = nn.Softmax(dim=-1)
        self.gumbel_connector = GumbelConnector(config.use_gpu)

    def get_embed(self, x):
        """
        Args:
            x: states
        Returns: d, a, s
        """
        # Todo: whether this one is needed? This common model?
        # [B, 392] -> [B, 256]
        common_h = self.e_common(x)
        # [B, 256] -> [B, 64] * 3
        d = self.e_1(common_h)
        a = self.e_2(common_h)
        s = self.e_3(common_h)
        # get the loss over here.
        input_encoder = torch.cat((d, a, s), dim = -1)


        _, hidden = self.encoder_rnn(input_encoder)

        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(x.size(0), self.embedding_size * self.hidden_factor)

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
        action_pred = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)
        return action_pred

    def forward(self, state, action):
        """
        Args:
            state:
            action:
        Returns: Decoding and Encoding bf states.
        """
        # common_h = self.e_common(state)
        # [B, 64] -> [B, seq_len, 64] = [B, 3, 64]
        d_e_1 = self.e_1(state).unsqueeze(1)
        a_e_1 = self.e_2(state).unsqueeze(1)
        s_e_1 = self.e_3(state).unsqueeze(1)
        # [B, 3, 64]

        # First classification
        domain_pred_1 = self.softmax(self.c_1(d_e_1))
        action_pred_1 = self.softmax(self.c_2(a_e_1))
        slot_pred_1 = self.softmax(self.c_3(s_e_1))

        encoder_input = torch.cat((d_e_1, a_e_1, s_e_1), dim = 1)
        whole, last_hidden = self.encoder_rnn(encoder_input)

        d_e_2 = whole[:, 0, :]
        a_e_2 = whole[:, 1, :]
        s_e_2 = whole[:, 2, :]

        # Second classification
        domain_pred_2 = self.softmax(self.c_1_last(d_e_2))
        action_pred_2 = self.softmax(self.c_2_last(a_e_2))
        slot_pred_2 = self.softmax(self.c_3_last(s_e_2))

        decoder_input = torch.cat((d_e_2, a_e_2, s_e_2), dim = -1)
        # action_h = self.c_whole(decoder_input)
        # hidden =
        last_hidden = last_hidden.transpose(1,0)
        last_hidden = last_hidden.view(state.size(0), -1)
        action_h = self.c_tiny(last_hidden.view())
        # get action and reconstruction
        action_whole = self.gumbel_connector(action_h, temperature = self.temperature, hard = False)
        recon_batch = self.decoder(decoder_input)


        return recon_batch, domain_pred_1, action_pred_1, slot_pred_1, domain_pred_2, action_pred_2, slot_pred_2, action_whole

# the failure one.
class AE_3_parallel_AE(BaseModel):
    def __init__(self, config):
        super(AE_3_parallel_AE, self).__init__(config)
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

        # (Embedding)_part_2
        self.e_2_1 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size*3, self.embedding_size),
            nn.LeakyReLU())

        self.e_2_2 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size*3, self.embedding_size),
            nn.LeakyReLU())

        self.e_2_3 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size*3, self.embedding_size),
            nn.LeakyReLU())

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

        self.c_1_last = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, self.embedding_size),

            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 17))

        self.c_2_last = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, self.embedding_size),

            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embedding_size, 34))

        self.c_3_last = nn.Sequential(
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
        # Todo: whether this one is needed? This common model?
        # [B, 392] -> [B, 256]

        d_e_1 = self.e_1(x)
        a_e_1 = self.e_2(x)
        s_e_1 = self.e_3(x)
        # get the whole embedding [domain, action, slot]
        e = torch.cat((d_e_1, a_e_1, s_e_1), dim = -1)

        d_e_2 = self.e_2_1(e)
        a_e_2 = self.e_2_2(e)
        s_e_2 = self.e_2_3(e)
        embedded_states = torch.cat((d_e_2, a_e_2, s_e_2), dim = -1)
        return embedded_states

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
        Returns:
        """
        fake_input = torch.cat((z_1, z_2, z_3), dim = -1)
        d = self.e_2_1(fake_input)
        a = self.e_2_2(fake_input)
        s = self.e_2_3(fake_input)
        fake_states = torch.cat((d, a, s), dim = -1)
        action_h = self.c_action(fake_states)
        fake_actions = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)
        return fake_states, fake_actions

    def forward(self, state, action):
        """
        Args:
            state:
            action:
        Returns: Decoding and Encoding bf states.
        """
        # common_h = self.e_common(state)
        # [B, 64] -> [B, seq_len, 64] = [B, 3, 64]
        d_e_1 = self.e_1(state)
        a_e_1 = self.e_2(state)
        s_e_1 = self.e_3(state)
        # [B, 3, 64]

        # First classification
        domain_pred_1 = self.softmax(self.c_1(d_e_1))
        action_pred_1 = self.softmax(self.c_2(a_e_1))
        slot_pred_1 = self.softmax(self.c_3(s_e_1))
        # get the whole embedding [domain, action, slot]
        e = torch.cat((d_e_1, a_e_1, s_e_1), dim = -1)

        d_e_2 = self.e_2_1(e)
        a_e_2 = self.e_2_2(e)
        s_e_2 = self.e_2_3(e)

        # Second classification
        domain_pred_2 = self.softmax(self.c_1_last(d_e_2))
        action_pred_2 = self.softmax(self.c_2_last(a_e_2))
        slot_pred_2 = self.softmax(self.c_3_last(s_e_2))

        decoder_input = torch.cat((d_e_2, a_e_2, s_e_2), dim = -1)
        # action_h = self.c_whole(decoder_input)
        action_h = self.c_action(decoder_input)
        # get action and reconstruction
        action_whole = self.gumbel_connector(action_h, temperature = self.temperature, hard = False)
        recon_batch = self.decoder(decoder_input)


        return recon_batch, domain_pred_1, action_pred_1, slot_pred_1, domain_pred_2, action_pred_2, slot_pred_2, action_whole

class AE_3_parallel_VAE(BaseModel):
    def __init__(self, config):
        super(AE_3_parallel_VAE, self).__init__(config)
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
        # self.e_1 = nn.Sequential(
        #     nn.Dropout(config.dropout),
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(),
        #     nn.Dropout(config.dropout),
        #     nn.Linear(128, self.embedding_size),
        #     nn.LeakyReLU())
        #
        # self.e_2 = nn.Sequential(
        #     nn.Dropout(config.dropout),
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(),
        #     nn.Dropout(config.dropout),
        #     nn.Linear(128, self.embedding_size),
        #     nn.LeakyReLU())
        #
        # self.e_3 = nn.Sequential(
        #     nn.Dropout(config.dropout),
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(),
        #     nn.Dropout(config.dropout),
        #     nn.Linear(128, self.embedding_size),
        #     nn.LeakyReLU())

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

        state_emb = torch.cat((mean_1, mean_2, mean_3), dim = -1)

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

        d_vae_e = self.latent2hidden_1(z_1)
        a_vae_e = self.latent2hidden_2(z_2)
        s_vae_e = self.latent2hidden_3(z_3)

        e_vae = torch.cat((d_vae_e, a_vae_e, s_vae_e), dim = -1)
        action_h = self.c_action(e_vae)
        action_pred_whole = self.gumbel_connector(action_h, temperature = self.config.gumbel_temp, hard = True)
        recon_batch = self.decoder(e_vae)
        # return all of this stuff.
        # get KL_loss
        d_kl_loss = (- 0.5 * torch.sum(1 + logv_1 - mean_1.pow(2) - logv_1.exp()))
        a_kl_loss = (- 0.5 * torch.sum(1 + logv_2 - mean_2.pow(2) - logv_2.exp()))
        s_kl_loss = (- 0.5 * torch.sum(1 + logv_3 - mean_3.pow(2) - logv_3.exp()))
        # Todo, chang this one to the seperate training.
        ELBO_whole = d_kl_loss + a_kl_loss + s_kl_loss
        return recon_batch, domain_pred, action_pred, slot_pred, action_pred_whole, ELBO_whole

class AE_3_seperate_VAE(BaseModel):
    def __init__(self, config):
        super(AE_3_seperate_VAE, self).__init__(config)
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
        self.latent2hidden_1=nn.Sequential(
                            nn.Dropout(config.dropout),
                            nn.Linear(self.embedding_size, 128),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(128, self.input_size))

        self.hidden2mean_2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.hidden2logv_2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.latent2hidden_2=nn.Sequential(
                            nn.Dropout(config.dropout),
                            nn.Linear(self.embedding_size, 128),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(128, self.input_size))

        self.hidden2mean_3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.hidden2logv_3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.latent2hidden_3=nn.Sequential(
                            nn.Dropout(config.dropout),
                            nn.Linear(self.embedding_size, 128),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(128, self.input_size))


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

        state_emb = torch.cat((mean_1, mean_2, mean_3), dim = -1)

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
        # TODO, figure out to use z or e to make the prediction.
        domain_pred = self.softmax(self.c_1(d_e))
        action_pred = self.softmax(self.c_2(z_2))
        slot_pred  =  self.softmax(self.c_3(z_3))

        recon_batch_d = self.latent2hidden_1(z_1)
        recon_batch_a = self.latent2hidden_2(z_2)
        recon_batch_s = self.latent2hidden_3(z_3)

        # return all of this stuff.
        # get KL_loss
        d_kl_loss = (- 0.5 * torch.sum(1 + logv_1 - mean_1.pow(2) - logv_1.exp()))
        a_kl_loss = (- 0.5 * torch.sum(1 + logv_2 - mean_2.pow(2) - logv_2.exp()))
        s_kl_loss = (- 0.5 * torch.sum(1 + logv_3 - mean_3.pow(2) - logv_3.exp()))
        return recon_batch_d, recon_batch_a, recon_batch_s, domain_pred, action_pred, slot_pred, d_kl_loss, a_kl_loss, s_kl_loss

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
        # Todo, should we use this one or mean to decode this stuff? I think use Z is a better choice.
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
