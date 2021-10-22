"""
数据集地址：http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
解压后放到./dataset文件夹中
"""
import re
from torch.utils.data import DataLoader, Dataset
import config
import os
import torch
import pickle


def tokenize(corpus):
    """
    文本内容预处理
    """

    # 去掉尖括号及其内容
    corpus = re.sub("<.*?>", " ", corpus)

    # 去掉其他符号
    filters = ['\\,', '\\:', '\\.', '\t', '\n', 'x97', 'x96', '#', '$', '%', '&']
    corpus = re.sub("|".join(filters), " ", corpus)

    # 分词
    tokens = [word.strip().lower() for word in corpus.split()]

    return tokens


class IMDBDataset(Dataset):
    def __init__(self, train=True):
        super(IMDBDataset, self).__init__()
        self.train_data_path = './dataset/aclImdb/train'
        self.test_data_path = './dataset/aclImdb/test'

        self.data_path = self.train_data_path if train else self.test_data_path

        """ 所有文件名放入列表 """
        all_data_path = [os.path.join(self.data_path, 'pos'), os.path.join(self.data_path, 'neg')]

        self.total_file_path = []

        for pos_neg_path in all_data_path:
            file_name_list = os.listdir(pos_neg_path)

            file_path_list = [
                os.path.join(pos_neg_path, i) for i in file_name_list if i.endswith('.txt')
            ]

            self.total_file_path.extend(file_path_list)

    def __getitem__(self, index):
        # 拿到一个文件的完整路径
        file_path = self.total_file_path[index]

        # 获取当前文本的标签
        label_str = file_path.split("\\")[-2]

        # 原标签为字符串，转为数字
        label = 0 if label_str == "neg" else 1

        # 获取文本内容
        content = open(file_path, encoding='utf-8').read()

        # 对文本内容进行分词
        data = tokenize(content)

        return data, label

    def __len__(self):
        return len(self.total_file_path)


def collate_fn(batch):
    """
    重写DataLoader的该方法，否则在DataLoader类中，default_collate会递归对batch个文本进行zip，导致对应句子分散
    :param ([tokens, label], [tokens, label] ... 一共batch个)
    """
    content, target = list(zip(*batch))  # zip(*)，解包

    # 单词 -> 数字
    if os.path.exists('./model/ws.pkl'):
        ws = pickle.load(open('./model/ws.pkl', 'rb'))
        content = [ws.word_to_sequence(i, max_len=config.max_len) for i in content]

    tokens = torch.LongTensor(content)
    label = torch.LongTensor(target)

    return tokens, label


def get_dataloader(train=True, batch_size=64, shuffle=True):
    return DataLoader(
        dataset=IMDBDataset(train=train), batch_size=batch_size,
        shuffle=shuffle, collate_fn=collate_fn
    )
