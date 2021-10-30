import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nhead, nclass, p_dropout):
        super(GAT, self).__init__()
        self.gat1 = GraphAttentionLayer(input_dim=nfeat, output_dim=nhid, num_heads=nhead, is_final_layer=False)
        self.gat2 = GraphAttentionLayer(input_dim=nhid * nhead, output_dim=nclass, num_heads=1, is_final_layer=True)
        self.p_dropout = p_dropout

    def forward(self, x, adj):
        x = F.elu(self.gat1(x, adj))
        x = F.dropout(x, self.p_dropout, training=self.training)
        x = self.gat2(x, adj)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    N = 5
    input_dim = 256
    hidden_dim = 512
    num_heads = 8
    n_class = 3
    dropout_prob = 0.6
    x = torch.rand(N, input_dim)
    adj_matrix = torch.tensor([[0, 1, 0, 1, 1],
                               [1, 0, 1, 0, 0],
                               [0, 1, 0, 1, 0],
                               [1, 0, 1, 0, 1],
                               [1, 0, 0, 1, 0]])
    gat = GAT(input_dim, hidden_dim, num_heads, n_class, dropout_prob)
    output = gat(x, adj_matrix)
    print(output.shape)
