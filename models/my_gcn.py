import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphConvolution
from torch_geometric.nn import GCNConv

# class myGCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(myGCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nhid)
#         self.cls = nn.Linear(nhid, nclass)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         # x = self.gc2(x, adj)
#         x = F.relu(self.gc2(x, adj))
#         embedding = F.dropout(x, self.dropout, training=self.training)
#         out = self.cls(embedding)
#         return embedding, F.log_softmax(out, dim=1)

class myGCN1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(myGCN1, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.cls = nn.Linear(nhid, nclass)
        self.dropout_p = dropout

    def forward(self, x, adj):
        edge_index = adj
        x = (self.conv1(x, edge_index))
        embedding = F.relu(x)
        out = self.cls(embedding)
        return embedding, F.log_softmax(out, dim=1)

class myGCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=2):
        super(myGCN2, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.cls = nn.Linear(nhid, nclass)
        self.dropout_p = dropout

    def forward(self, x, adj):
        x = self.conv1(x,adj)
        x = F.relu(x)
        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, adj)
        embedding = F.relu(x)
        out = self.cls(embedding)
        return embedding, F.log_softmax(out, dim=1)