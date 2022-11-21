import torch.nn as nn
import torch.nn.functional as F

class mlp2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2):
        super(mlp2, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.cls = nn.Linear(nhid, nclass)
        self.dropout_p = dropout
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.fc2(x)
        embedding = F.relu(x)
        out = self.cls(embedding)
        return embedding, F.log_softmax(out, dim=1)