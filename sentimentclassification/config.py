import torch

batch_size = 64
embedding_size = 100
max_len = 200
hidden_size = 128
num_layers = 2
dropout = 0.4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
