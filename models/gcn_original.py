'''
copy from Kipf https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
'''
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphConvolution
from torch_geometric.nn import GCNConv

# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return F.log_softmax(x, dim=1)

class GCN1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(GCN1, self).__init__()
        self.conv1 = GCNConv(nfeat, nclass)

    def forward(self, x, adj):
        edge_index = adj
        x = (self.conv1(x, edge_index))
        return F.log_softmax(x, dim=1)

class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=2):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        self.dropout_p = dropout

    def forward(self, x, adj):
        x = self.conv1(x,adj)
        x = F.relu(x)
        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)