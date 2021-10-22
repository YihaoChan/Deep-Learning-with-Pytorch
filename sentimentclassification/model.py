import torch
import torch.nn as nn
import config
import torch.nn.functional as F

"""
---双向LSTM模型结构---
1. 原始数据：X: [batch_size, seq_len] 一共有多少个句子，一个句子有几个单词
2. 嵌入层nn.Embedding(input_size=batch_size, embedding_dim=config.embedding_size)：
    送入多少个句子，每个单词要嵌入为一个多长的行向量
3. 前向传播嵌入之后的维度：[batch_size, seq_len, config.embedding_size]，作为LSTM层的输入
4. LSTM层nn.LSTM(
    input_size=config.embedding_size, hidden_size=config.hidden_size,
    num_layers=config.num_layer, bidirectional=config.bidirectional, batch_first=True
   )  
5. 前向传播后三个输出：
   output: [batch_size, seq_len, hidden_size * directions]
   h_t: [num_layers * directions, batch_size, hidden_size]
   c_t: [num_layers * directions, batch_size, hidden_size]  
6. 获取两个方向最后一次的输出进行连接：
   output_fw = h_t[-2, :, :]
   output_bw = h_t[-1, :, :]
   output = torch.cat([output_fw, output_bw], dim=-1) -> [batch_size, hidden_size * directions]    
7. 全连接层nn.Linear(
    hidden_size * directions, output_size
   )  
"""


class SentiClassModel(nn.Module):
    def __init__(self, ws_len):
        super(SentiClassModel, self).__init__()
        self.embedding_size = config.embedding_size

        self.embedding = nn.Embedding(ws_len, config.embedding_size)

        self.lstm = nn.LSTM(
            input_size=config.embedding_size, hidden_size=config.hidden_size,
            num_layers=config.num_layers, bidirectional=True,
            batch_first=True, dropout=config.dropout
        )

        self.fc = nn.Linear(config.hidden_size * 2, 2)

    def forward(self, input):
        # input必须为LongTensor
        x = self.embedding(input)  # -> [batch_size, seq_len(已经用max_len限制), embed_size]

        # out: [batch_size, seq_len, hidden_size * directions]
        # h, c: [num_layer * directions, batch_size, hidden_size]
        out, (h_t, c_t) = self.lstm(x)

        # 获取两个方向最后一次的输出进行连接
        output_fw = h_t[-2, :, :]
        output_bw = h_t[-1, :, :]
        output = torch.cat([output_fw, output_bw], dim=-1)  # [batch_size, hidden_size * 2]

        # 送入二维数据
        out = self.fc(output)

        return F.log_softmax(out, dim=-1)

