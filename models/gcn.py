""" Some parts of code are referenced from https://github.com/tkipf/pygcn."""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import scipy.sparse as sp
import numpy as np
from .resnet import resnet18
from utils.utils import get_glove_embeddings

import warnings
warnings.filterwarnings("ignore")


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.resnet_dim = 512
        self.action_space = args.action_space
        self.hidden_state_dim = args.hidden_state_dim
        self.n_objects = args.n_objects
        self.glove_embeddings = get_glove_embeddings(args)

        self.glove_embed = nn.Linear(args.glove_dim, args.glove_embed_dim)
        self.action_embed = nn.Linear(self.action_space, args.action_dim)
        self.fused_embed = nn.Linear(self.resnet_dim + args.glove_embed_dim + args.action_dim + args.gcn_dim, args.hidden_state_dim)
        self.lstm = nn.LSTM(args.hidden_state_dim, args.hidden_state_dim, bidirectional = False, batch_first = True)
        
        self.actor_linear = nn.Sequential(
            nn.Linear(args.hidden_state_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, args.action_space),
            nn.ReLU()
        )

        self.critic_linear = nn.Sequential(
            nn.Linear(args.hidden_state_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(p = args.dropout_rate)

        # get and normalize adjacency matrix.
        A_raw = torch.load(args.adjmat_dir)
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A))

        # glove embeddings for all the objs.
        objects = open(args.obj_file_dir).readlines()
        objects = [o.strip() for o in objects]
        all_glove = torch.zeros(args.n_objects, args.glove_dim)
        # self.glove_embeddings = h5py.File(args.glove_file, "r")
        # for i in range(args.n_objects):
        #     all_glove[i, :] = torch.Tensor(self.glove_embeddings[objects[i]][:])
        for i in range(args.n_objects):
            all_glove[i, :] = torch.Tensor(self.glove_embeddings[objects[i]])
        self.all_glove = nn.Parameter(all_glove)
        self.all_glove.requires_grad = False

        self.get_word_embed = nn.Linear(args.glove_dim, args.glove_embed_dim)
        #  self.get_class_embed = nn.Linear(1000, 512)

        self.W0 = nn.Linear(self.resnet_dim + args.glove_embed_dim, 256, bias = False)
        self.W1 = nn.Linear(256, 256, bias = False)
        self.W2 = nn.Linear(256, 1, bias = False)
        self.final_mapping = nn.Linear(args.n_objects, args.gcn_dim)

        self.resnet = resnet18(pretrained = True, freeze = True, model_path = args.ResNet18_path)


    def gcn_embed(self, input_, bs, len_seq):
        # bs x len_seq x (shape)
        if self.args.features_or_images == False:
            input_ = self.resnet(input_)

        class_embed = input_.squeeze(2).squeeze(2).unsqueeze(1).repeat(1, self.n_objects, 1)
        word_embed = self.get_word_embed(self.all_glove.detach())
        word_embed = word_embed.unsqueeze(0).repeat((len_seq, 1, 1)).unsqueeze(0).repeat((bs, 1, 1, 1))
        word_embed = word_embed.view(bs * len_seq, *word_embed.shape[2:])
        x = torch.cat((class_embed, word_embed), dim = 2) # (bs x len_seq) x self.n_object x 768
        
        res = None
        for i in range(x.shape[0]):
            x_i = x[i]
            x_i = torch.mm(self.A, x_i)
            x_i = F.relu(self.W0(x_i))
            x_i = torch.mm(self.A, x_i)
            x_i = F.relu(self.W1(x_i))
            x_i = torch.mm(self.A, x_i)
            x_i = F.relu(self.W2(x_i))
            x_i = x_i.view(1, self.n_objects)
            x_i = self.final_mapping(x_i)
            if res == None:
                res = x_i
            else:
                res = torch.cat((res, x_i), dim = 0)
        # print(f'res:{res.shape}')

        return res


    def embedding(self, target, input_, action_probs):
        # bs x len_seq x (shape)
        bs, len_seq = target.shape[0], target.shape[1]
        input_ = input_.view(bs * len_seq, *input_.shape[2:])
        if self.args.features_or_images == False:
            input_ = self.resnet(input_)
        
        target = target.view(bs * len_seq, *target.shape[2:])
        action_probs = action_probs.view(bs * len_seq, *action_probs.shape[2:])

        glove_embedding = F.relu(self.glove_embed(target))
        action_embedding = F.relu(self.action_embed(action_probs))
        image_embedding = self.dropout(input_).squeeze(2).squeeze(2)
        gcn_embedding  = self.gcn_embed(input_, bs, len_seq)
        fused_embedding = torch.cat((image_embedding, glove_embedding, action_embedding, gcn_embedding), dim = 1)
        out = F.relu(self.fused_embed(fused_embedding))
        out = out.view(bs, len_seq, -1)  # bs x len_seq x 512

        return out

    
    def forward(self, target_embedding, input_, action_probs, h0 = None, c0 = None):
        # bs x len_seq x (shape)
        feature_embedding = self.embedding(target_embedding, input_, action_probs)
        bs, len_seq = input_.shape[0], input_.shape[1]
        if h0 == None:
            h0 = torch.zeros((1, bs, self.hidden_state_dim), device = input_.device)
        if c0 == None:
            c0 = torch.zeros((1, bs, self.hidden_state_dim), device = input_.device)

        state_feature, (hs, cs) = self.lstm(feature_embedding, (h0, c0)) # bs x len_seq x 512
        state_feature = state_feature.reshape(bs * len_seq, -1)
        
        actor_out = self.actor_linear(state_feature)
        critic_out = self.critic_linear(state_feature)
        actor_out = actor_out.view(bs, len_seq, -1)
        critic_out = critic_out.view(bs, len_seq, -1)

        return actor_out, critic_out, (hs, cs)