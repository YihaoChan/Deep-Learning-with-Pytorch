import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # shape: [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape: [max_len, 1]
        # 1 / 10000^(2i/d_model) 用指数形式表达 i：维度
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # PE(pos, 2i) [:, 0::2]表示从0开始到最后步长为2，即代表偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # PE(pos, 2i+1) [:, 1::2]表示从1开始到最后步长为2，即代表奇数位置

        pe = pe.unsqueeze(0).transpose(0, 1)  # shape: [max_len, 1, d_model]

        self.register_buffer('pe', pe)  # 不更新这个参数

    def forward(self, x):
        """
        x.shape: [max_len, batch_size, d_model] 已经经过word embedding，d_model就是word embedding的维度
        """
        # 词向量和位置编码直接相加
        x = x + self.pe[:x.shape[0], :, :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = (softmax(QK^T / 根号d_k)) * V
    此处属于单头层，被多头调用，所以维度上有heads
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch, heads, len_q, d_k]
        :param K: [batch, heads, len_k, d_k]
        :param V: [batch, heads, len_v(=len_k), d_v]
        :param attn_mask: 在计算QKV时，要把mask处置为负无穷 [batch, heads, seq_len, seq_len]
                          句子单词的下三角矩阵，所以维度是[seq_len, seq_len]
        """
        # Q * K^T / 根号d_k
        # [batch, heads, len_q, d_k] * [batch, heads, d_k, len_k] => [batch, heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.shape[-1])
        scores.masked_fill_(attn_mask, -float('inf'))  # 把现有向量需要做掩码(已经设为pad)的地方设置为负无穷，再做softmax

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch, heads, len_q, d_v]

        return context, attn_mask


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention + Add&Norm
    """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 对最后一维做concatenate，所以线性层输出为d_x * heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.W_O = nn.Linear(d_v * n_heads, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        """
        输入的x，当做Q、K、V的初始化，然后分别乘上W_Q、W_K、W_V
        :param Q: [batch, len_q, d_model]
        :param K: [batch, len_k, d_model]
        :param V: [batch, len_v(=len_k), d_model]
        :param attn_mask: [batch, len_q, len_k]
        """
        residual = Q  # 残差结构的x就是没有经过前向传播的东西，在这里就是Q
        batch_size = Q.shape[0]

        q_i = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch, heads, len_q, d_k]
        k_i = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch, heads, len_k, d_k]
        v_i = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # [batch, heads, len_v, d_v]

        # 把pad信息重复了n个头
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # [batch, heads, len_q, len_k]

        context, attn_mask = ScaledDotProductAttention()(q_i, k_i, v_i, attn_mask)  # [batch, heads, len_q, d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_heads * d_v)  # [batch, len_q, heads * d_v]

        output = self.W_O(context)  # [batch, len_q, d_model]
        return self.layer_norm(residual + output), attn_mask


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward + Add&Norm
    """

    def __init__(self):
        super(PositionWiseFeedForward, self).__init__()
        # 如果要接BN/LN操作，最好是不设置偏置，因为不起作用，而且占显存
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.fc(x)
        return self.layer_norm(output + x)


class EncoderLayer(nn.Module):
    """
    一个Encoder Layer由Multi-Head和Feed-Forward两个sub-layer组成
    """

    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.multi_head = MultiHeadAttention()
        self.ffn = PositionWiseFeedForward()

    def forward(self, encoder_inputs, attn_mask):
        """
        用三个矩阵将原始输入往不同的向量空间进行投影
        """
        context, attn_mask = self.multi_head(encoder_inputs, encoder_inputs, encoder_inputs, attn_mask)
        output = self.ffn(context)
        return output, attn_mask


def get_attn_pad_mask(q, k):
    """
    :param q: Q向量
    :param k: K向量和V向量
    Q、K不一定一样长
    """
    batch_size, len_q = q.size()
    batch_size, len_k = k.size()
    # eq(zero) is PAD token
    pad_attn_mask = k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k] one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)  # 源语言嵌入
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, encoder_inputs):
        """
        :param encoder_inputs: [batch_size, src_len]
        """
        encoder_outputs = self.embedding(encoder_inputs)  # [batch_size, src_len, d_model]
        encoder_outputs = self.pe(encoder_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]

        attn_mask = get_attn_pad_mask(encoder_inputs, encoder_inputs)  # [batch_size, len_q, len_k]
        self_attns = []
        for encoder_layer in self.layers:
            encoder_outputs, self_attn = encoder_layer(encoder_outputs, attn_mask)  # 上一层编码层的输出作为下一层的输入
            self_attns.append(self_attn)

        return encoder_outputs, self_attns  # 编码器的输出，编码器自注意力相乘并掩码的矩阵


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.decoder_self_attn = MultiHeadAttention()  # 解码器自己的自注意力
        self.decoder_encoder_attn = MultiHeadAttention()  # 解码器的Q和编码器送过来的K、V做交互注意力
        self.ffn = PositionWiseFeedForward()

    def forward(self, decoder_inputs, encoder_outputs, decoder_self_attn_mask, decorder_encoder_attn_mask):
        """
        :param decoder_inputs: 解码器输入，为Ground Truth
        :param encoder_outputs: 编码器输出，作为Decoder阶段的K、V
        :param decoder_self_attn_mask: 解码器自己对于Ground Truth目标语言的QKV自注意力掩码
        :param decorder_encoder_attn_mask: Decoder和Encoder的交互掩码
        """
        decoder_outputs, decoder_self_attn = self.decoder_self_attn(decoder_inputs,  # 解码器自己的Q
                                                                    decoder_inputs,  # 解码器自己的K
                                                                    decoder_inputs,  # 解码器自己的V
                                                                    decoder_self_attn_mask)  # 自注意力的掩码
        dec_outputs, dec_enc_attn = self.decoder_encoder_attn(decoder_outputs,  # 解码器Ground Truth作为Q
                                                              encoder_outputs,  # 编码器送过来的K
                                                              encoder_outputs,  # 编码器送过来的V
                                                              decorder_encoder_attn_mask)  # 交互掩码
        dec_outputs = self.ffn(dec_outputs)
        return dec_outputs, decoder_self_attn, dec_enc_attn


def get_attn_subsequent_mask(seq):
    """
    Teacher Forcing，mask掉t时刻后面的词，即：得到一个上三角矩阵
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)  # 目标语言嵌入
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, decoder_inputs, encoder_inputs, encoder_outputs):
        decoder_outputs = self.embedding(decoder_inputs)  # [batch_size, tgt_len, d_model]
        decoder_outputs = self.pe(decoder_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(decoder_inputs, decoder_inputs)  # [batch_size, len_q, len_k]

        # 这个做的是自注意层的mask部分，就是当前单词之后看不到，使用一个上三角为1的矩阵
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        # 两个矩阵相加，大于0的为1，不大于0的为0，为1的在之后就会被fill到-inf，在计算softmax时这部分就为0
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        # 交互注意力机制中的mask矩阵，enc的输入是k，去看这个k里面哪些是pad符号，给到后面的模型
        dec_enc_attn_mask = get_attn_pad_mask(decoder_inputs, encoder_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for decoder_layer in self.layers:
            decoder_outputs, dec_self_attn, dec_enc_attn = decoder_layer(decoder_outputs,  # 上一层的输出作为下一层的输入
                                                                         encoder_outputs,
                                                                         dec_self_attn_mask,
                                                                         dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return decoder_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, encoder_inputs, decoder_inputs):
        """
        :param encoder_inputs: [batch_size, src_len]
        :param decoder_inputs: [batch_size, tgt_len]
        """
        encoder_outputs, encoder_self_attns = self.encoder(encoder_inputs)
        decoder_outputs, decoder_self_attns, decoder_encoder_attns = self.decoder(decoder_inputs,
                                                                                  encoder_inputs,
                                                                                  encoder_outputs)
        decoder_logits = self.fc(decoder_outputs)  # [batch_size, tgt_len, tgt_vocab_size]
        return decoder_logits.view(-1, decoder_logits.size(-1)), \
               encoder_self_attns, \
               decoder_self_attns, \
               decoder_encoder_attns


if __name__ == '__main__':
    # S: Symbol that shows starting of decoding input
    # E: Symbol that shows end of decoding output
    # P: Symbol that will fill in blank sequence if current batch data size is shorter than time steps
    """
                      enc_input              dec_input           dec_output
                       原始数据              ground truth          模型输出
    """
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    """
    1. 编码阶段捕捉原始待翻译序列中的词和词之间的关系，即德语句子中词和词之间的关系
    2. Encoder的输出作为解码阶段的K和V，而Ground Truth即英语句子为解码阶段的input，这个英语句子也要进行一次Q、K、V的计算
    3. 英语句子的Q、K、V计算结果作为Q，和Encoder传到解码阶段的K、V进行计算，相当于让英语句子去学习德语句子的词与词之间的关系
       然后输出，和标签做loss
    """

    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5  # length of source
    tgt_len = 5  # length of target

    d_model = 512  # word embedding维度和位置编码维度都是d_model
    d_ff = 2048  # FeedForward层维度
    n_heads = 8  # 8个head
    d_k = d_v = d_model // n_heads  # d_k = d_v = d_model / heads
    n_layers = 6  # Encoder由多少层堆叠

    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    def make_batch(sentences):
        input_batch = [[src_vocab[n] for n in sentences[0].split()]]
        output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
        target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
        return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(20):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()


    def showgraph(attn):
        attn = attn[-1].squeeze(0)[0]
        attn = attn.squeeze(0).data.numpy()
        fig = plt.figure(figsize=(n_heads, n_heads))  # [n_heads, n_heads]
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attn, cmap='viridis')
        ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
        ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
        plt.show()


    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns)

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns)

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns)
