from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18


class BaseModel(torch.nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.resnet_dim = 512
        self.action_space = args.action_space
        self.hidden_state_dim = args.hidden_state_dim

        self.glove_embed = nn.Linear(args.glove_dim, args.glove_embed_dim)
        self.action_embed = nn.Linear(self.action_space, args.action_dim)
        self.fused_embed = nn.Linear(self.resnet_dim + args.glove_embed_dim + args.action_dim, args.hidden_state_dim)
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
        self.resnet = resnet18(pretrained = True, freeze = True, model_path = args.ResNet18_path)


    def embedding(self, target, input_, action_probs):
        # bs x len_seq x (shape)
        bs, len_seq = target.shape[0], target.shape[1]
        input_ = input_.view(bs * len_seq, *input_.shape[2:])
        if self.args.features_or_images == False:
            input_ = self.resnet(input_)
        target = target.view(bs * len_seq, *target.shape[2:])
        # print('action_probs shape:', action_probs.shape)
        action_probs = action_probs.view(bs * len_seq, *action_probs.shape[2:])

        glove_embedding = F.relu(self.glove_embed(target))
        action_embedding = F.relu(self.action_embed(action_probs))
        image_embedding = self.dropout(input_).squeeze(2).squeeze(2)
        
        # print('shape:', image_embedding.shape, glove_embedding.shape, action_embedding.shape)
        fused_embedding = torch.cat((image_embedding, glove_embedding, action_embedding), dim = 1)
        out = F.relu(self.fused_embed(fused_embedding))
        out = out.view(bs, len_seq, -1)  # bs x len_seq x 512

        return out


    def forward(self, target_embedding, input_, action_probs, h0 = None, c0 = None):
        # bs x len_seq x (shape)
        # print('input_shape:', input_.shape)
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
        # hs = hs.view(bs, len_seq, -1)
        # cs = cs.view(bs, len_seq, -1)

        return actor_out, critic_out, (hs, cs)