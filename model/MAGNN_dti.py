import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from model.base_MAGNN_dti import MAGNN_ctr_ntype_specific
from model.GNN import myGAT
from dgl.nn.pytorch import GraphConv, GATConv

def adj_normalize(adj):
    rowsum = adj.sum(1)
    r_inv = torch.pow(rowsum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    adj_ = r_mat_inv * adj * r_mat_inv
    return adj_

def MinMax_scalar(x):
    min = x.min(1).values
    min = min.view(-1, 1).repeat(1, x.shape[1])
    max = x.max(1).values
    max = max.view(-1, 1).repeat(1, x.shape[1])
    scalar = (x - min) / (max - min)
    return scalar

def normalize(x):
    rowsum = x.sum(1)
    rowsum = rowsum.view(-1, 1).repeat(1, x.shape[1])
    x_norm = x / rowsum
    return x_norm


# for link prediction task
class MAGNN_lp_layer(nn.Module):
    def __init__(self,
                 edge_dim,
                 etypes_num1,
                 etypes_num2,
                 in_dims1,
                 in_dims2,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 decode):
        
        super(MAGNN_lp_layer, self).__init__()

        self.num_layers = num_layers

        self.register_buffer('epsilon', torch.tensor([1e-12], dtype=torch.float32))

        self.head = heads
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(num_hidden, num_hidden, self.head[0], feat_drop, attn_drop,negative_slope, residual, activation))
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(num_hidden*self.head[l-1], num_hidden, self.head[l], feat_drop, attn_drop,negative_slope, residual, activation))
        
        self.user_layer = myGAT(edge_dim,
                                etypes_num1,
                                in_dims1,
                                num_hidden,
                                num_classes,
                                num_layers,
                                heads,
                                activation,
                                feat_drop,
                                attn_drop,
                                negative_slope,
                                residual,
                                alpha,
                                decode)
        self.item_layer = myGAT(edge_dim,
                                etypes_num2,
                                in_dims2,
                                num_hidden,
                                num_classes,
                                num_layers,
                                heads,
                                activation,
                                feat_drop,
                                attn_drop,
                                negative_slope,
                                residual,
                                alpha,
                                decode)

    def l2_norm(self, x):
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
            return x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))


    def forward(self, inputs):
        g1, g2, features_list1, features_list2, e_feat1, e_feat2, left, right, mid = inputs
        
        h_user_left, h_user_right= self.user_layer(g1, features_list1, e_feat1, left, right, mid)
        h_item_left, h_item_right = self.item_layer(g2, features_list2, e_feat2, left, right, mid)
        
        return [h_user_left, h_item_right]


class GCN_layer(nn.Module):
    def __init__(self, dim, mp_ls=None):
        super(GCN_layer, self).__init__()
        self.gcn1 = nn.Parameter(torch.zeros([dim, 128]), requires_grad=True)
        # self.gcn2 = nn.Parameter(torch.zeros([128, 2]), requires_grad=True)
        ###########
        self.gcn2 = nn.Parameter(torch.zeros([128, 256]), requires_grad=True)
        self.linear2 = nn.Linear(256,2)
        ###########
        nn.init.xavier_normal_(self.gcn1, gain=1.414)
        nn.init.xavier_normal_(self.gcn2, gain=1.414)

    def forward(self, x, adj):
        # x = MinMax_scalar(x)
        # x = normalize(x)
        adj = F.softmax(torch.matmul(x, x.T), dim=-1)
        # adj = adj_normalize(adj)
        x = F.relu(torch.matmul(torch.matmul(adj, x), self.gcn1))
        # x = F.leaky_relu(torch.matmul(torch.matmul(adj, x), self.gcn1), negative_slope=0.01)
        # x = torch.matmul(torch.matmul(adj, x), self.gcn2)
        ############
        x = F.relu(torch.matmul(torch.matmul(adj, x), self.gcn2))
        # x = F.leaky_relu(torch.matmul(torch.matmul(adj, x), self.gcn2), negative_slope=0.01)
        x = self.linear2(x)
        ############

        return x


class linear_module(nn.Module):
    def __init__(self, dim):
        super(linear_module, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim/2), bias=True)
        self.fc2 = nn.Linear(int(dim/2), 2, bias=False)
        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, x, x2=None):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class MAGNN_lp(nn.Module):
    def __init__(self,
                 edge_dim,
                 etypes_num1,
                 etypes_num2,
                 in_dims1,
                 in_dims2,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 decode,
                 args=None):
        super(MAGNN_lp, self).__init__()
        self.args = args
        self.heads = heads

        self.layer1 = MAGNN_lp_layer(edge_dim,
                                     etypes_num1,
                                     etypes_num2,
                                     in_dims1,
                                     in_dims2,
                                     num_hidden,
                                     num_classes,
                                     num_layers,
                                     heads,
                                     activation,
                                     feat_drop,
                                     attn_drop,
                                     negative_slope,
                                     residual,
                                     alpha,
                                     decode)
        dim = num_hidden * (len(heads)+1)
        
        if self.args.semantic_fusion == 'concatenation':
            dim = num_hidden * (len(heads)+1) * 2
        # predictor
        if self.args.predictor =='gcn':
            self.classifier = GCN_layer(dim=dim)
        elif self.args.predictor == 'linear':
            self.classifier = linear_module(dim=dim)

    def forward(self, inputs):
        g1, g2, features_list1, features_list2, e_feat1, e_feat2, left, right, mid, adj = inputs

        [h_user, h_item] = self.layer1((g1, g2, features_list1, features_list2, e_feat1, e_feat2, left, right, mid))
        x = torch.cat([h_user, h_item], dim=1)
        x_out = self.classifier(x, adj)   
        return F.softmax(x_out, dim=-1)
