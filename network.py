#!/usr/bin/python3

# import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask = mask.type(dtype)
    return mask


class Net(nn.Module):

    def __init__(self, input_dim, hidden_size, n_classes, n_recipe,
                 recipe_weight=1.0, no_gru=False):
        super(Net, self).__init__()

        self.n_classes = n_classes
        self.n_recipe = n_recipe
        self.recipe_weight = recipe_weight
        self.no_gru = no_gru

        # removing gru layer
        if not no_gru:  # if we train using gru i.e no_gru=False i.e default case
            self.primary_gru = nn.GRU(input_dim, hidden_size // 2, 1, bidirectional=True, batch_first=True)
            self.action_attention_layer = nn.Linear(hidden_size, n_classes)
        else:  # remove gru i.e no_gru=True
            self.action_attention_layer = nn.Linear(input_dim, n_classes)
            hidden_size = input_dim

        # randomly zero out some of the elements of the input tensor with probability p, during training p ==> p_drop not p_keep
        # self.dropout = nn.Dropout(p=0.5)

        self.action_matrix = nn.Parameter(torch.randn(n_classes, hidden_size))
        self.action_bias = nn.Parameter(torch.randn(1, n_classes))
        self.__init_weight__(self.action_matrix, self.action_bias)

        self.recipe_attention_matrix = nn.Parameter(torch.randn(n_recipe, hidden_size))
        self.recipe_attention_bias = nn.Parameter(torch.randn(1, n_recipe, 1))
        self.__init_weight__(self.recipe_attention_matrix, self.recipe_attention_bias)

        self.recipe_matrix = nn.Parameter(torch.randn(n_recipe, hidden_size))
        self.recipe_bias = nn.Parameter(torch.randn(1, n_recipe))
        self.__init_weight__(self.recipe_matrix, self.recipe_bias)

    def __init_weight__(self, matrix, bias):
        torch.nn.init.xavier_normal_(matrix)
        torch.nn.init.xavier_normal_(bias)

    def forward(self, seq, seq_len):
        ### GRU
        # removing gru layer
        if not self.no_gru:
            packed = pack_padded_sequence(seq, seq_len, batch_first=True, enforce_sorted=False)
            feature_packed, dummy = self.primary_gru(packed)  # batch_size, sequence_len, feature
            feature, _ = pad_packed_sequence(feature_packed, batch_first=True)
            self.primary_feature = feature  # self.primary_feature is an output of gru. to be fed into following layers
        else:  # not including gru
            self.primary_feature = seq

        # old feature: seq: B, T, 64   also hidden_size is 64  when you run action_attention_layer, it works
        # new feature: seq: B, T, 1024      hidden_size is 64

        ### action classification # directly put the input features into the attention layer
        # apply attention

        # change seq to self.primary_feature
        B, T, H = self.primary_feature.shape
        feature_flat = self.primary_feature.view(B * T, H)

        # apply dropout # dropout data only in train mode -not test
        # feature_flat = self.dropout(feature_flat)

        atn_logit = self.action_attention_layer(feature_flat)
        self.atn_logit = atn_logit = atn_logit.view(B, T, self.n_classes)

        # softmax but mask out padded elements
        mask = sequence_mask(seq_len, maxlen=seq.shape[1], dtype=torch.float).unsqueeze(-1)
        exp_logit = torch.exp(self.atn_logit) * mask  # B, T, H
        sum_ = exp_logit.sum(dim=1, keepdim=True)
        temporal_attention = exp_logit / sum_
        # temporal_attention is of the shape/dimension batch_size x sequence/number of frames x number of actions/classes i.e B,T,C
        # temporal_attention gives me the attn for all actions and all frames for all videos
        # get the attention vector for action1 for the first video: p(a_1) for video1 = temporal_attention[0, :, 0]
        # network's output: attention vector p(a_1) of action a_1

        # in order to get the attn vector for the first action in the first video

        # temporal_attention[0] gives me first video not batch i.e gives me the attn for all actions and all frames for video1
        # temporal_attention gives me the attn for all actions and all frames for all videos

        self.temporal_attention_output = temporal_attention

        temporal_attention = temporal_attention.unsqueeze(-1)  # B, T, C, 1
        # action_feature = seq.unsqueeze(2) # B, T, 1, H      # for no gru
        action_feature = self.primary_feature.unsqueeze(2)  # B, T, 1, H
        action_feature = temporal_attention * action_feature  # B, T, C, H
        action_feature = action_feature.sum(1)  # B, C, H

        # classify
        self.action_logit = torch.einsum("bch,ch->bc", action_feature, self.action_matrix) + self.action_bias
        # same as create a classifier layer for each action
        # equals to the following pseudo code

        # action_classifier = [ nn.Linear(hidden_size, 1) for a in range(num_actions) ]
        # action_logit = []
        # for a in range(num_actions):
        #     classifier = action_classifier[a]
        #     logit = classifier(action_feature[:, a, :])
        #     action_logit.append(logit)

        #### recipe classification
        # apply attention
        recipe_atn_logit = torch.einsum("bch,rh->brc", action_feature,
                                        self.recipe_attention_matrix) + self.recipe_attention_bias
        # similar to the action classifier
        # but create an attention layer for each recipe

        # recipe_attention_layer = [ nn.Linear(hidden_size, 1) for r in range(num_recipe) ]
        # recipe_atn_logit = []
        # for r in range(num_recipe):
        #     layer = recipe_attention_layer[a]
        #     atn_logit = layer(action_feature[:, a, :]) # B, C, H -> B, C, 1
        #     recipe_atn_logit.append(atn_logit)

        recipe_atn_logit = nn.functional.softmax(recipe_atn_logit, dim=2).unsqueeze(-1)  # B, R, C, 1
        recipe_feature = action_feature.unsqueeze(1)  # B, 1, C, H
        recipe_feature = recipe_feature * recipe_atn_logit  # B, R, C, H
        recipe_feature = recipe_feature.sum(2)  # B, R, H

        # classify
        self.recipe_logit = torch.einsum("brh,rh->br", recipe_feature, self.recipe_matrix) + self.recipe_bias
        # exactly the same as action classifier

        return self.action_logit, self.recipe_logit, self.temporal_attention_output

    def compute_loss(self, recipe, action, pos_action_weight=None):
        # import ipdb; ipdb.set_trace()
        self.action_loss = nn.functional.binary_cross_entropy_with_logits(
            self.action_logit, action, weight=pos_action_weight)  # shape B, C

        recipe_logprob = nn.functional.log_softmax(self.recipe_logit, dim=1)
        self.recipe_loss = nn.functional.nll_loss(recipe_logprob, recipe)  # shape B, R

        self.loss = self.recipe_weight * self.recipe_loss + self.action_loss

        ### compute accuracy
        self.action_acc = ((self.action_logit > 0).float() == action).float()
        self.action_acc = self.action_acc[:, 1:].mean().item()  # ignore bg

        self.action_pred = self.action_logit > 0

        recipe_pred = self.recipe_logit.argmax(1)
        self.recipe_acc = (recipe_pred == recipe).float().mean().item()

    def forward_and_loss(self, seq, seq_len, action, recipe, pos_action_weight=None):
        self.forward(seq, seq_len)
        self.compute_loss(recipe, action, pos_action_weight=pos_action_weight)

    def save_model(self, network_file):
        torch.save(self.state_dict(), network_file)  # save network's parameters instead of entire net

    # restore learned parameters from one (trained) net to another i.e load model
    # def load_model(self, network_file):
    #     net = Net(self, input_dim, hidden_size, n_classes, n_recipe,
    #                 recipe_weight=1.0, no_gru=False)
    #     net.load_state_dict(torch.load(network_file)) # deserialize PATH to saved net into a dict object before passing to load_state_dict()
    #     net.eval()







