#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author:
# @Date  : 2021/11/1 16:16
# @Desc  :
import os.path
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_set import DataSet
from gcn_conv import GCNConv
from utils import BPRLoss, EmbLoss

def contrastive_loss(embeds1, embeds2, temp):
    embeds1 = F.normalize(embeds1 + 1e-8, p=2)
    embeds2 = F.normalize(embeds2 + 1e-8, p=2)

    nume = torch.exp(torch.sum(embeds1 * embeds2, dim=-1) / temp)
    deno = torch.exp(embeds1 @ embeds2.T / temp).sum(-1) + 1e-8
    return -torch.log(nume / deno).mean()


class GraphEncoder(nn.Module):
    def __init__(self, layer_nums, hidden_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim, add_self_loops=False, cached=False) for i in range(layer_nums)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        result = [x]
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index)
            x = F.normalize(x, dim=-1)
            result.append(x / (i + 1))
        result = torch.stack(result, dim=0)
        result = torch.sum(result, dim=0)
        return result


class Mutual_Attention(nn.Module):

    def __init__(self, input_dim, dim_qk, dim_v):
        super(Mutual_Attention, self).__init__()
        self._norm_fact = 1 / sqrt(dim_qk)

    def forward(self, q_token, k_token, v_token):
        att = nn.Softmax(dim=-1)(torch.matmul(q_token, k_token.transpose(-1, -2)) * self._norm_fact)
        att = torch.matmul(att, v_token)
        return att


class HGNNLayer(nn.Module):
    def __init__(self):
        super(HGNNLayer, self).__init__()
        self.leaky = 0.5
        self.act = nn.LeakyReLU(negative_slope=self.leaky)

    def forward(self, adj, embeds):
        lat = adj.T @ embeds
        ret = adj @ lat
        return ret


class Hyper_behavior_gcn(nn.Module):
    def __init__(self, args, n_users, n_items, layer_nums=2):
        super(Hyper_behavior_gcn, self).__init__()
        self.args = args
        self.leaky = 0.5
        self.layer_nums = layer_nums
        #self.keepRate = 1 - args.hyper_dropout
        self.act = nn.LeakyReLU(negative_slope=self.leaky)
        latdim = args.embedding_size
        hyperNum = args.hyper_nums
        self.n_users = n_users
        self.n_items = n_items
        self.hgnnLayer = HGNNLayer()
        self.uHyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(latdim, hyperNum)))
        self.iHyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(latdim, hyperNum)))
        self.dropout1 = nn.Dropout(p=args.hyper_dropout)
        self.dropout2 = nn.Dropout(p=args.hyper_dropout)

    def forward(self, embeds):
        embeds = embeds.detach()
        lats = [embeds]

        user_embedding, item_embedding = torch.split(lats[-1], [self.n_users + 1, self.n_items + 1])
        uuHyper = user_embedding @ self.uHyper
        iiHyper = item_embedding @ self.iHyper

        for i in range(self.layer_nums):
            user_embedding, item_embedding = torch.split(lats[-1], [self.n_users + 1, self.n_items + 1])
            hyperULat = self.hgnnLayer(self.dropout1(uuHyper), user_embedding)
            hyperILat = self.hgnnLayer(self.dropout2(iiHyper), item_embedding)
            hyper_all = torch.cat([hyperULat, hyperILat], dim=0)
            lats.append(F.normalize(hyper_all, dim=-1))
        embeds = sum(lats) / len(lats)
        return embeds


class GRU_black(nn.Module):
    def __init__(self, embedd_dim=64):
        super(GRU_black, self).__init__()
        self.embedd_dim = embedd_dim
        self.W_x_r = nn.Linear(embedd_dim, embedd_dim, bias=False)
        self.W_h_r = nn.Linear(embedd_dim, embedd_dim, bias=False)
        self.b_r = nn.Parameter(nn.init.xavier_normal_(torch.zeros(size=(1, self.embedd_dim))))

        self.W_x_z = nn.Linear(embedd_dim, embedd_dim, bias=False)
        self.W_h_z = nn.Linear(embedd_dim, embedd_dim, bias=False)
        self.b_z = nn.Parameter(nn.init.xavier_normal_(torch.zeros(size=(1, self.embedd_dim))))

        self.W_x_h = nn.Linear(embedd_dim, embedd_dim, bias=False)
        self.W_h_h = nn.Linear(embedd_dim, embedd_dim, bias=False)
        self.b_h = nn.Parameter(nn.init.xavier_normal_(torch.zeros(size=(1, self.embedd_dim))))

    def forward(self, global_embedding: torch.Tensor, curr_behavior_embedding: torch.Tensor):
        R_t = torch.sigmoid(self.W_h_r(global_embedding) + self.W_x_r(curr_behavior_embedding) + self.b_r)
        Z_t = torch.sigmoid(self.W_h_z(global_embedding) + self.W_x_z(curr_behavior_embedding) + self.b_z)
        H_tile = torch.tanh(self.W_x_h(curr_behavior_embedding) + self.W_h_h(torch.mul(R_t, global_embedding)) + self.b_h)
        H_t = torch.mul(Z_t, global_embedding) + torch.mul((1 - Z_t), H_tile)
        return H_t, H_tile


class HEC_GCN(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(HEC_GCN, self).__init__()
        self.args = args
        self.layers_nums = self.args.layers_nums
        self.cl_coefficient = self.args.cl_coefficient
        self.loss_coefficient = self.args.loss_coefficient
        self.device = args.device
        self.layers = args.layers
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index
        self.all_edge_index = dataset.all_edge_index
        self.item_behaviour_degree = dataset.item_behaviour_degree.to(self.device)
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.behavior_gru = nn.ModuleDict({behavior: GRU_black(self.embedding_size) for behavior in self.behaviors})
        self.behavior_graph_encoder = nn.ModuleDict({behavior: GraphEncoder(layer_nums=self.layers_nums[idx], hidden_dim=self.embedding_size, dropout=self.node_dropout) for idx, behavior in enumerate(self.behaviors)})

        self.global_graph_encoder = GraphEncoder(self.layers, self.embedding_size, self.node_dropout)
        self.W = nn.Parameter(torch.ones(len(self.behaviors)))
        self.behavior_hyper_graph_encoder = nn.ModuleDict({behavior: Hyper_behavior_gcn(args=args, n_users=self.n_users, n_items=self.n_items, layer_nums=1) for idx, behavior in enumerate(self.behaviors)})


        self.dim_qk = args.dim_qk
        self.dim_v = args.dim_v
        self.attention_user = Mutual_Attention(self.embedding_size, self.dim_qk, self.dim_v)
        self.attention_item = Mutual_Attention(self.embedding_size, self.dim_qk, self.dim_v)

        self.reg_weight = args.reg_weight
        self.layers = args.layers
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model
        self.message_dropout = nn.Dropout(p=args.message_dropout)

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self.apply(self._init_weights)

        self._load_model()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def behaivor_gcn_propagate(self, total_embeddings):
        """
        gcn propagate in each behavior
        """
        temp_total_embeddings = total_embeddings
        all_user_embeddings, all_item_embeddings = [], []
        all_user_hyper_embeddings, all_item_hyper_embeddings = [], []
        all_behavior_all_embedding_dict: dict = {}
        for behavior in self.behaviors:
            indices = self.edge_index[behavior].to(self.device)
            behavior_embeddings = self.behavior_graph_encoder[behavior](total_embeddings, indices)
            behavior_hyper_embeddings = self.behavior_hyper_graph_encoder[behavior](behavior_embeddings)
            all_behavior_all_embedding_dict[behavior] = [total_embeddings, behavior_embeddings, behavior_hyper_embeddings]
            total_embeddings = (temp_total_embeddings + behavior_embeddings + behavior_hyper_embeddings) / 3

            user_embedding, item_embedding = torch.split(total_embeddings, [self.n_users + 1, self.n_items + 1])
            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)



        all_user_embeddings = torch.stack(all_user_embeddings, dim=1)
        all_item_embeddings = torch.stack(all_item_embeddings, dim=1)
        return all_user_embeddings, all_item_embeddings, all_behavior_all_embedding_dict

    def global_gcn(self, total_embeddings, indices):
        total_embeddings = self.global_graph_encoder(total_embeddings, indices.to(self.device))

        return total_embeddings

    def forward(self, batch_data):
        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = self.global_gcn(all_embeddings, self.all_edge_index)

        user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])
        all_user_embeddings, all_item_embeddings, all_behavior_all_embedding_dict = self.behaivor_gcn_propagate(all_embeddings)

        all_user_embeddings = self.attention_user(all_user_embeddings, all_user_embeddings, all_user_embeddings)
        all_user_embeddings = all_user_embeddings + user_embedding.unsqueeze(1)

        ##add: item adaptive
        all_item_embeddings = self.attention_item(all_item_embeddings, all_item_embeddings, all_item_embeddings)
        all_item_embeddings = all_item_embeddings + item_embedding.unsqueeze(1)

        total_loss = 0
        cl_loss = 0
        for i, behavior in enumerate(self.behaviors):
            data = batch_data[:, i]
            users = data[:, 0].long()
            items = data[:, 1:].long()

            [total_embeddings, behavior_embeddings, behavior_hyper_embeddings] = all_behavior_all_embedding_dict[behavior]
            user_behavior_embeddings, item_behavior_embeddings = torch.split(behavior_embeddings, [self.n_users + 1, self.n_items + 1])
            user_hyper_behavior_embeddings, item_hyper_behavior_embeddings = torch.split(behavior_hyper_embeddings, [self.n_users + 1, self.n_items + 1])
            user_global_embeddings, item_global_embeddings = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])
            tau = self.args.tau
            cl_user_loss = (
                  self.cl_coefficient[0] * contrastive_loss(user_behavior_embeddings[torch.unique(users)], user_hyper_behavior_embeddings[torch.unique(users)], temp=tau)  ## A
                + self.cl_coefficient[1] * contrastive_loss(user_behavior_embeddings[torch.unique(users)], user_global_embeddings[torch.unique(users)], temp=tau)  ## B
                + self.cl_coefficient[2] * contrastive_loss(user_hyper_behavior_embeddings[torch.unique(users)], user_global_embeddings[torch.unique(users)], temp=tau)  ## C
            )
            cl_item_loss = (
                  self.cl_coefficient[0] * contrastive_loss(item_behavior_embeddings[torch.unique(items)], item_hyper_behavior_embeddings[torch.unique(items)], temp=tau)  ## A
                + self.cl_coefficient[1] * contrastive_loss(item_behavior_embeddings[torch.unique(items)], item_global_embeddings[torch.unique(items)], temp=tau)  ## B
                + self.cl_coefficient[2] * contrastive_loss(item_hyper_behavior_embeddings[torch.unique(items)], item_global_embeddings[torch.unique(items)], temp=tau) ## C
            )
            cl_loss += cl_user_loss + cl_item_loss

            user_feature = all_user_embeddings[:, i][users.view(-1, 1)]
            item_feature = all_item_embeddings[:, i][items]
            scores = torch.sum(user_feature * item_feature, dim=2)
            total_loss += self.loss_coefficient[i] * self.bpr_loss(scores[:, 0], scores[:, 1])
        total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight) + self.args.alpha * cl_loss

        return total_loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            all_embeddings = self.global_gcn(all_embeddings, self.all_edge_index)

            user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])
            all_user_embeddings, all_item_embeddings, _ = self.behaivor_gcn_propagate(all_embeddings)

            target_user_embeddings = all_user_embeddings[:, -1].unsqueeze(1)
            target_user_embeddings = self.attention_user(target_user_embeddings, all_user_embeddings, all_user_embeddings)
            self.storage_user_embeddings = target_user_embeddings.squeeze() + user_embedding
            ##add: item adaptive
            target_item_embeddings = all_item_embeddings[:, -1].unsqueeze(1)
            target_item_embeddings = self.attention_item(target_item_embeddings, all_item_embeddings, all_item_embeddings)
            self.storage_item_embeddings = target_item_embeddings.squeeze() + item_embedding
        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings.transpose(0, 1))

        return scores
