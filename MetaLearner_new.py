from util import *

import random
import torch
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F


# social embedding: prediction
class MetaLearner(torch.nn.Module):
    def __init__(self, config):
        super(MetaLearner, self).__init__()
        self.use_cuda = config['use_cuda']
        self.config = config

        # prediction parameters
        self.vars = torch.nn.ParameterDict()
        self.embd_dim = config['embedding_dim']
        # True:1  False:0
        self.social_types = 2 + int(self.config['use_coclick'])
        self.social_encoder = SocialEncoder(config, self.vars, 'social')
        # self.vars['social_merge'] = self.get_initialed_para_matrix(1, self.social_types)

        self.gcn_module = TRPN(config, self.vars)
        # self.gcn_module = TRPN(n_feat=self.config['embedding_dim'], n_queries=self.config['num_ways'] * 1)
        self.bce_loss = nn.BCELoss()
        self.num_tasks = self.config['num_tasks']
        self.num_ways = self.config['num_ways']
        self.num_shots = self.config['num_shots']
        self.num_queries = self.config['num_queries']

        self.dim_1 = self.embd_dim * (config['use_fea_item'] + 2)
        self.dim_2 = config['first_fc_hidden_dim']
        self.dim_3 = config['second_fc_hidden_dim']
        self.dropout = config['dropout']

        self.vars['ml_fc_w1'] = self.get_initialed_para_matrix(self.dim_2, self.dim_1)
        self.vars['ml_fc_b1'] = self.get_zero_para_bias(self.dim_2)

        self.vars['ml_fc_w2'] = self.get_initialed_para_matrix(self.dim_3, self.dim_2)
        self.vars['ml_fc_b2'] = self.get_zero_para_bias(self.dim_3)

        self.vars['ml_fc_w3'] = self.get_initialed_para_matrix(1, self.dim_3)
        self.vars['ml_fc_b3'] = self.get_zero_para_bias(1)

        self.vars['ml_social_w1'] = self.get_initialed_para_matrix(self.embd_dim, self.embd_dim)
        self.vars['ml_social_b1'] = self.get_zero_para_bias(self.embd_dim)
        self.vars['ml_social_w2'] = self.get_initialed_para_matrix(1, self.embd_dim)

    def get_initialed_para_matrix(self, out_num, in_num):
        w = torch.nn.Parameter(torch.ones([out_num, in_num]))
        torch.nn.init.xavier_normal_(w)
        return w

    def get_zero_para_bias(self, num):
        return torch.nn.Parameter(torch.zeros(num))

    def forward(self, item_emb, user_emb, vars_dict=None, **kwargs):
        if vars_dict is None:
            vars_dict = self.vars

        x_i = item_emb
        x_u = user_emb

        preference = self.get_user_social_preference(vars_dict, **kwargs)
        x_self = preference[0].repeat(x_i.shape[0], 1)
        x_social = preference[1].repeat(x_i.shape[0], 1)
        trpn_loss = preference[2]
        x = torch.cat((x_i, x_self, x_social), 1)

        x = torch.tanh(F.linear(x, vars_dict['ml_fc_w1'], vars_dict['ml_fc_b1']))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = torch.tanh(F.linear(x, vars_dict['ml_fc_w2'], vars_dict['ml_fc_b2']))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.linear(x, vars_dict['ml_fc_w3'], vars_dict['ml_fc_b3'])
        return x.squeeze(), trpn_loss

    def get_task_embd(self, **kwargs):
        preference = self.get_user_social_preference(self.vars,
                                                     **kwargs)
        task_embd = torch.cat((preference[0], preference[1]), dim=1)
        return task_embd

    def get_task_batch(self, user_data, seed=None):
        if seed is not None:
            random.seed(seed)

        data_size = [self.embd_dim]
        num_tasks = self.num_tasks
        num_ways = self.num_ways
        num_shots = self.num_shots
        num_queries = self.num_queries
        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):  # 25
            data = np.zeros(shape=[num_tasks] + data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):  # 5
            data = np.zeros(shape=[num_tasks] + data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        full_class_list = list(user_data.keys())
        label_list = list(range(0, 5))  # [0, 1, 2, 3, 4]
        random.shuffle(label_list)
        """对于一个任务先从所有的类里面随机取N_way个类，
        然后从N_way的每个类中随机取num_shots + num_queries张图片去分别构建支持集和查询集"""
        # for each task   #20
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)
            # for each sampled class in task  #5
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(user_data[task_class_list[c_idx]], num_shots + num_queries)

                # load sample for support set  #5
                for i_idx in range(num_shots):
                    # set data   #tt.arg.features为false所以执行 else语句
                    support_data[i_idx + c_idx * num_shots][t_idx] = class_data_list[i_idx]
                    support_label[i_idx + c_idx * num_shots][t_idx] = label_list[c_idx]

                # load sample for query set  #1个
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = class_data_list[num_shots + i_idx]
                    query_label[i_idx + c_idx * num_queries][t_idx] = label_list[c_idx]

        support_data = torch.stack([torch.from_numpy(data).float() for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float() for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(query_data[i]).float() for i in label_list], 1)
        query_label = torch.stack([torch.from_numpy(query_label[i]).float() for i in label_list], 1)

        return [support_data, support_label, query_data, query_label]

    def get_user_social_preference(self, vars_dict=None, **kwargs):
        if vars_dict is None:
            vars_dict = self.vars

        self_users, self_items, self_mask = kwargs['self_users_embd'], kwargs['self_items_embd'], kwargs['self_mask']
        self_embd_wl_list, social_embd_wl_list = [], []

        social_users, social_items, social_mask = kwargs['social_users_embd'], kwargs['social_items_embd'], kwargs[
            'social_mask']
        self_embd_wl, social_embd_wl = self.social_encoder(self_users, self_items, self_mask, social_users,
                                                           social_items, social_mask, vars_dict)
        # x_self=torch.Size([5, 1, 32]) x_social=torch.Size([5, n, 32])
        self_embd_wl_list.append(self_embd_wl)
        social_embd_wl_list.append(self_embd_wl)  # 将自己也放进来
        social_embd_wl_list.append(social_embd_wl)

        implicit_users, implicit_items, implicit_mask = kwargs['implicit_users_embd'], kwargs['implicit_items_embd'], \
                                                        kwargs['implicit_mask']
        self_embd_wl, social_embd_wl = self.social_encoder(self_users, self_items, self_mask, implicit_users,
                                                           implicit_items, implicit_mask, vars_dict)

        self_embd_wl_list.append(self_embd_wl)
        social_embd_wl_list.append(social_embd_wl)

        if self.config['use_coclick']:
            coclick_users, coclick_items, coclick_mask = kwargs['coclick_users_embd'], kwargs['coclick_items_embd'], \
                                                         kwargs['coclick_mask']
            self_embd_wl, social_embd_wl = self.social_encoder(self_users, self_items, self_mask, coclick_users,
                                                               coclick_items, coclick_mask, vars_dict)
            self_embd_wl_list.append(self_embd_wl)  # torch.Size([5, 1, 32])
            social_embd_wl_list.append(social_embd_wl)  # torch.Size([5, n, 32])

        # print("social_embd_wl_list",social_embd_wl_list,len(social_embd_wl_list))  #[[5,n1,32],[5,n2,32]]
        # print("x_self_list",x_self_list,len(x_self_list))
        # x_self=torch.mean(self_embd_wl_list[0], dim=0).reshape(1, 32)
        self_embd = self_embd_wl_list[0]
        x_self = self.aggregate_rate(torch.squeeze(self_embd, dim=1), vars_dict)
        x_social = torch.cat(social_embd_wl_list, 1)  # 按照第二个维度拼接  #[5,n,32]

        user_data = dict()
        user_data["1"] = x_social[0, :, :].tolist()
        user_data["2"] = x_social[1, :, :].tolist()
        user_data["3"] = x_social[2, :, :].tolist()
        user_data["4"] = x_social[3, :, :].tolist()
        user_data["5"] = x_social[4, :, :].tolist()

        num_ways = self.num_ways
        num_shots = self.num_shots
        meta_batch_size = self.num_tasks
        embd_dim = self.embd_dim

        num_supports = num_ways * num_shots
        num_queries = num_ways * 1
        num_samples = num_supports + num_queries

        support_edge_mask = torch.zeros(meta_batch_size, num_samples, num_samples)
        support_edge_mask[:, :num_supports, :num_supports] = 1

        evaluation_mask = torch.ones(meta_batch_size, num_samples, num_samples)

        for c in range(num_ways):  # 5
            evaluation_mask[:, ((c + 1) * num_shots):(c + 1) * num_shots, :num_supports] = 0
            evaluation_mask[:, :num_supports, ((c + 1) * num_shots):(c + 1) * num_shots] = 0

        support_data, support_label, query_data, query_label = self.get_task_batch(user_data)
        full_data = torch.cat([support_data, query_data], 1)
        full_label = torch.cat([support_label, query_label], 1)
        num_samples = full_label.size(1)
        label_i = full_label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)
        edge = torch.eq(label_i, label_j).float()
        edge = edge.unsqueeze(1)
        edge = torch.cat([edge, 1 - edge], 1)  # [1,2,30,30]
        full_edge = edge
        init_edge = full_edge.clone()

        init_edge[:, :, num_supports:, :] = 0.5
        init_edge[:, :, :, num_supports:] = 0.5

        for i in range(num_queries):
            init_edge[:, 0, num_supports + i, num_supports + i] = 1.0  # 查询集


            init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

        for c in range(num_ways):
            init_edge[:, :, ((c + 1) * num_shots):(c + 1) * num_shots, :num_supports] = 0.5
            init_edge[:, :, :num_supports, ((c + 1) * num_shots):(c + 1) * num_shots] = 0.5
        ##torch.Size([30, 192])
        wl_embd, learned_score_list, query_score_list = self.gcn_module(node_feat=full_data,
                                                                        adj=init_edge[:, 0, :num_supports,
                                                                            :num_supports])

        loss1_pos = (self.bce_loss(learned_score_list, full_edge[:, 0, :, :]) * full_edge[:, 0, :,
                                                                                :] * evaluation_mask).sum() / (
                        (evaluation_mask * full_edge[:, 0, :, :]).sum())
        loss1_neg = (self.bce_loss(learned_score_list, full_edge[:, 0, :, :]) * (
                    1 - full_edge[:, 0, :, :]) * evaluation_mask).sum() / (
                        (evaluation_mask * (1. - full_edge[:, 0, :, :])).sum())
        loss2_pos = (self.bce_loss(query_score_list, full_edge[:, 0, num_supports:, :]) * full_edge[:, 0, num_supports:,
                                                                                          :] * evaluation_mask[:,
                                                                                               num_supports:,
                                                                                               :]).sum() / (
                        (evaluation_mask[:, num_supports:, :] * full_edge[:, 0, num_supports:, :]).sum())
        loss2_neg = (self.bce_loss(query_score_list, full_edge[:, 0, num_supports:, :]) * (
                    1. - full_edge[:, 0, num_supports:, :]) * evaluation_mask[:, num_supports:, :]).sum() / (
                        (evaluation_mask[:, num_supports:, :] * (1. - full_edge[:, 0, num_supports:, :])).sum())

        total_loss = (loss1_pos + loss1_neg + loss2_pos + loss2_neg) / 4

        social_embd = wl_embd[:num_supports, :embd_dim].reshape(num_ways, num_ways, embd_dim)
        att = torch.bmm(self_embd, social_embd.transpose(1, 2)).softmax(dim=2)
        social_embd = torch.bmm(att, social_embd)
        x_social = self.aggregate_rate(torch.squeeze(social_embd, dim=1), vars_dict)

        return x_self, x_social, total_loss

    def aggregate_rate(self, embd, vars_dict):
        embd_trans = torch.relu(F.linear(embd, vars_dict['ml_social_w1'], vars_dict['ml_social_b1']))
        att = torch.mm(vars_dict['ml_social_w2'], embd_trans.transpose(0, 1)).softmax(dim=1)
        embd = torch.mm(att, embd)
        return embd

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars

    def get_parameter_size(self):
        parameter_name_size = dict()
        for key in self.vars.keys():
            weight_size = np.prod(self.vars[key].size())
            parameter_name_size[key] = weight_size

        return parameter_name_size

    def get_user_preference_embedding(self, items_emb, users_embd, mask):
        item_embd_agg = items_emb * torch.unsqueeze(mask, dim=2)
        item_embd_agg = torch.sum(item_embd_agg, dim=1)
        mask_len = torch.sum(mask, dim=1, keepdim=True)
        user_preference_embd = torch.div(item_embd_agg, mask_len)
        return user_preference_embd


# social embedding: embedding
class SocialEncoder(torch.nn.Module):
    def __init__(self, config, vars_dict, name):
        super(SocialEncoder, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.vars = vars_dict
        self.name = name
        self.embd_dim = config['embedding_dim']  # 32
        used_feat_num = config['use_fea_user'] + config['use_fea_item']  # 1+1=2
        self.vars['ml_user_w'] = self.get_initialed_para_matrix(self.embd_dim, self.embd_dim * used_feat_num)
        self.vars['ml_user_b'] = self.get_zero_para_bias(self.embd_dim)
        self.social_types = 2 + int(self.config['use_coclick'])

    def forward(self, self_users, self_items, self_mask, social_users, social_items, social_mask, vars_dict=None):
        if vars_dict is None:
            vars_dict = self.vars
        self_embd_agg = self.aggregate_items(self_users, self_items,
                                             self_mask)
        self_embd = F.relu(
            F.linear(self_embd_agg, vars_dict['ml_user_w'], vars_dict['ml_user_b']))
        self_embd_wl = self_embd

        if social_users is None:
            social_embd = self_embd
            social_embd_wl = social_embd
        else:
            social_embd_agg = self.aggregate_items(social_users, social_items,
                                                   social_mask)
            social_embd = F.relu(
                F.linear(social_embd_agg, vars_dict['ml_user_w'], vars_dict['ml_user_b']))
            social_embd_wl = social_embd

        return self_embd_wl, social_embd_wl

    def aggregate_items(self, users, items, mask):
        users_items = torch.cat((users, items), dim=-1) * torch.unsqueeze(mask,
                                                                          dim=3)
        users_items_sum = torch.sum(users_items, dim=2)
        mask_len = torch.sum(mask, dim=2, keepdim=True) + 0.0001
        users_items_agg = torch.div(users_items_sum,
                                    mask_len)
        return users_items_agg

    def get_initialed_para_matrix(self, out_num, in_num):
        w = torch.nn.Parameter(torch.ones([out_num, in_num]))
        torch.nn.init.xavier_normal_(w)
        return w

    def get_zero_para_bias(self, num):
        return torch.nn.Parameter(torch.zeros(num))

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars


class GraphConvolution(nn.Module):
    def __init__(self, config,vars_dict,bias=True):
        super(GraphConvolution, self).__init__()
        self.config = config
        self.vars = vars_dict
        self.in_features = config['embedding_dim']*(config['num_ways']+1)
        self.out_features = config['embedding_dim']*(config['num_ways']+1)

        self.vars['weight_G'] = self.get_initialed_para_matrix(self.in_features, self.out_features)
        if bias:
            self.vars['bias_G'] = self.get_zero_para_bias(self.out_features)
        self.reset_parameters()

    def reset_parameters(self,vars_dict=None):
        if vars_dict is None:
            vars_dict = self.vars
        stdv = 1. / math.sqrt(vars_dict['weight_G'].size(1))
        vars_dict['weight_G'].data.uniform_(-stdv, stdv)
        if vars_dict['bias_G'] is not None:
            vars_dict['bias_G'].data.uniform_(-stdv, stdv)

    def forward(self, input, adj,vars_dict=None):
        if vars_dict is None:
            vars_dict = self.vars
        # print('device:', input.device, self.weight.device)
        support = torch.mm(input, vars_dict['weight_G'])
        output = torch.spmm(adj, support)
        if vars_dict['bias_G'] is not None:
            return output + vars_dict['bias_G']
        else:
            return output

    def get_initialed_para_matrix(self, out_num, in_num):
        w = torch.nn.Parameter(torch.ones([out_num, in_num]))
        torch.nn.init.xavier_normal_(w)
        return w
    def get_zero_para_bias(self, num):
        return torch.nn.Parameter(torch.zeros(num))
    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()
    def update_parameters(self):
        return self.vars




class TRPN(nn.Module):
    def __init__(self, config, vars_dict):
        super(TRPN, self).__init__()
        self.config = config

        self.vars = vars_dict
        self.dim_1 = config['embedding_dim']  # 32
        self.dim_2 = config['hidden_layers0']  # 640
        self.dim_3 = config['hidden_layers1']  # 320
        self.dim_4 = config['hidden_layers2']  # 160

        self.vars['trpn_fc1_w1'] = self.get_initialed_para_matrix(self.dim_2, self.dim_1*2)  #
        self.vars['trpn_fc1_b1'] = self.get_zero_para_bias(self.dim_2)

        self.vars['trpn_fc1_w2'] = self.get_initialed_para_matrix(self.dim_3, self.dim_2)
        self.vars['trpn_fc1_b2'] = self.get_zero_para_bias(self.dim_3)

        self.vars['trpn_fc1_w3'] = self.get_initialed_para_matrix(1, self.dim_3)
        self.vars['trpn_fc1_b3'] = self.get_zero_para_bias(1)

        self.vars['trpn_fc2_w1'] = self.get_initialed_para_matrix(self.dim_3, self.dim_1*6)  #
        self.vars['trpn_fc2_b1'] = self.get_zero_para_bias(self.dim_3)

        self.vars['trpn_fc2_w2'] = self.get_initialed_para_matrix(self.dim_4, self.dim_3)
        self.vars['trpn_fc2_b2'] = self.get_zero_para_bias(self.dim_4)

        self.vars['trpn_fc2_w3'] = self.get_initialed_para_matrix(5, self.dim_4)
        self.vars['trpn_fc2_b3'] = self.get_zero_para_bias(5)

        self.gc = GraphConvolution(config,self.vars)

    def get_initialed_para_matrix(self, out_num, in_num):
        w = torch.nn.Parameter(torch.ones([out_num, in_num]))
        torch.nn.init.xavier_normal_(w)
        return w

    def get_zero_para_bias(self, num):
        return torch.nn.Parameter(torch.zeros(num))

    def forward(self, node_feat, adj, vars_dict=None):
        if vars_dict is None:
            vars_dict = self.vars

        num_tasks = node_feat.size(0)
        num_samples = node_feat.size(1)
        num_supports = adj.size(1)
        num_queries = num_samples - num_supports
        in_features_2 = node_feat.size(2) * 2

        x_i = node_feat.unsqueeze(2).repeat(1, 1, node_feat.size(1), 1)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.cat((x_i, x_j), -1)

        gcn_input_feat = node_feat

        for i in range(num_queries):
            gcn_input_feat = torch.cat(
                (gcn_input_feat, node_feat[:, num_supports + i, :].unsqueeze(1).repeat(1, num_samples, 1)),-1)

        learned_score_list = []
        query_score_list = []
        wl_embd_list = []
        for i in range(num_tasks):
            x = x_ij[i].contiguous().view(num_samples ** 2, in_features_2)  # [900,64]
            x = F.relu(F.linear(x, vars_dict['trpn_fc1_w1'], vars_dict['trpn_fc1_b1']))
            x = F.relu(F.linear(x, vars_dict['trpn_fc1_w2'], vars_dict['trpn_fc1_b2']))
            x = F.linear(x, vars_dict['trpn_fc1_w3'], vars_dict['trpn_fc1_b3'])
            x = torch.sigmoid(x)
            learned_score = x.view(num_samples, num_samples)  # torch.Size([30, 30])

            learned_adj = learned_score.clone()
            ones = torch.ones(learned_adj[:num_supports, :num_supports].size())  # torch.Size([25, 25])
            learned_adj[:num_supports, :num_supports] = torch.where(adj[i] > 0, ones, -learned_adj[:num_supports,
                                                                                       :num_supports])

            wl_embd = F.relu(self.gc(gcn_input_feat[i], learned_adj))  # torch.Size([30, 192])
            x = wl_embd
            x = F.relu(F.linear(x, vars_dict['trpn_fc2_w1'], vars_dict['trpn_fc2_b1']))
            x = F.relu(F.linear(x, vars_dict['trpn_fc2_w2'], vars_dict['trpn_fc2_b2']))
            x = F.linear(x, vars_dict['trpn_fc2_w3'], vars_dict['trpn_fc2_b3'])
            query_score = torch.sigmoid(x)

            wl_embd_list.append(wl_embd)
            learned_score_list.append(learned_score)
            query_score_list.append(query_score)

        return torch.mean(torch.stack(wl_embd_list, 0), dim=0), torch.stack(learned_score_list, 0), torch.stack(
            query_score_list, 0).transpose(1, 2)

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars


