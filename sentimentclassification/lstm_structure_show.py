import torch
import torch.nn as nn

"""
---双向LSTM模型结构---
1. 原始数据：X: [batch_size, seq_len] 一共有多少个句子，一个句子有几个单词
2. 嵌入层nn.Embedding(input_size=batch_size, embedding_dim=config.embedding_size)：
    送入多少个句子，每个单词要嵌入为一个多长的行向量
3. 嵌入之后的维度：[batch_size, seq_len, config.embedding_size]，作为LSTM层的输入
4. LSTM层nn.LSTM(
    input_size=config.embedding_size, hidden_size=config.hidden_size,
    num_layers=config.num_layer, bidirectional=config.bidirectional, batch_first=True
   )  
5. 三个输出：
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

# batch: 16 embedding_dim:100 hidden_size:32 seq_len:5
x = torch.rand([16, 5]).long()

embed = nn.Embedding(16, 100)

embedded = embed(x)

lstm = nn.LSTM(100, 32, batch_first=True, num_layers=2, bidirectional=True)

out, (h, c) = lstm(embedded)

print('原始数据维度：', x.shape)
print('嵌入后的维度：', embedded.shape)
print('LSTM层out维度：', out.shape)
print('LSTM层h_t维度：', h.shape)
print('LSTM层c_t维度：', c.shape)

# 最后一个时间步的输出，取到seq_len
last_output = out[:, -1, :]
print('最后一个时间步的输出的维度：', last_output.shape)

# 获取最后一层的hidden_state，取到num_layer
last_hidden_state_fw = h[-1, :, :]
last_hidden_state_bw = h[-2, :, :]
print('最后一层的hidden_state维度：', last_hidden_state_fw.shape)
print('倒数第二层的hidden_state维度：', last_hidden_state_bw.shape)

bidirection_hidden = torch.cat([last_hidden_state_fw, last_hidden_state_bw], dim=-1)
# [batch_size, hidden_size * 2] 因为cat
print('连接之后的维度：', bidirection_hidden.shape)
