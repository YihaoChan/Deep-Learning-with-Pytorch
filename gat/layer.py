import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, is_final_layer):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim  # 论文中的F
        self.output_dim = output_dim  # 论文中的F'
        self.num_heads = num_heads  # 多头
        self.is_final_layer = is_final_layer  # 如果是最后一个GAT层，就不能concatenate，应该average

        self.W = nn.Parameter(torch.empty(input_dim, output_dim))  # W.shape: (input_dim, output_dim)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(2 * output_dim, 1))  # a.shape: (2*output_dim, 1)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, inputs, adj_matrix):
        """
        :param inputs: 论文中的hi
        :param adj_matrix: 邻接矩阵，用于计算邻居注意力和掩码 shape: (N, N)
        """
        multihead_outputs = []

        for _ in range(self.num_heads):
            # inputs就是hi，实现W*hi、W*hj
            Wh = torch.mm(inputs, self.W)  # (N, input_dim) * (input_dim, output_dim) => (N, output_dim)

            # Messages (h_neigh and h_self) are [extracted from x] => 从原始矩阵中提出出来做初始化，后面再学习

            # (W*hi) * a = (N, output_dim) * (2*output_dim / 2, 1) => (N, 1)
            attention_self = torch.mm(Wh, self.a[:self.output_dim, :])
            # (W*hj) * a = (N, output_dim) * (2*output_dim / 2, 1) => (N, 1)
            attention_nbr = torch.mm(Wh, self.a[self.output_dim:, :])

            # a^T * [W*hi || W*hj] = a^T * (W*hi) + a^T * (W*hj) = ((W*hi) * a) + ((W*hj) * a)
            # 把维度变成(N, N)，才能结合邻接矩阵做mask
            attention_pure = attention_self + attention_nbr.T  # 广播机制
            attention_pure = F.leaky_relu(attention_pure)
            attention_pure = F.log_softmax(attention_pure, dim=1)

            # inject the graph stucture for mask => alpha_ij
            mask = -10e9 * torch.ones_like(attention_pure)
            a_ij = torch.where(adj_matrix > 0, attention_pure, mask)  # (N, N)

            # ===========================================================================
            # 这一步非常非常关键，否则loss会爆炸！！！
            a_ij = F.softmax(a_ij, dim=1)
            # ===========================================================================

            output = torch.mm(a_ij, Wh)  # (N, N) * (N, output_dim) => (N, output_dim)

            multihead_outputs.append(output)

        if self.is_final_layer:
            # average
            final_output = torch.mean(torch.stack(multihead_outputs, dim=1), dim=1)  # (N, output_dim)
        else:
            # concatenate
            final_output = torch.cat(multihead_outputs, dim=1)  # (N, heads * output_dim)

        return final_output


if __name__ == '__main__':
    N = 5
    input_dim = 256
    output_dim = 10
    x = torch.rand(N, input_dim)
    adj_matrix = torch.tensor([[0, 1, 0, 1, 1],
                               [1, 0, 1, 0, 0],
                               [0, 1, 0, 1, 0],
                               [1, 0, 1, 0, 1],
                               [1, 0, 0, 1, 0]])

    # 非最后一层
    gat_layer_not_final = GraphAttentionLayer(input_dim=input_dim,
                                              output_dim=output_dim,
                                              num_heads=18,  # 多头
                                              is_final_layer=False)
    output_not_final = gat_layer_not_final(x, adj_matrix)
    print("not final:", output_not_final.shape)

    # 最后一层
    gat_layer_final = GraphAttentionLayer(input_dim=input_dim,
                                          output_dim=output_dim,
                                          num_heads=1,  # 单头
                                          is_final_layer=True)
    output_final = gat_layer_final(x, adj_matrix)
    print("final:", output_final.shape)
