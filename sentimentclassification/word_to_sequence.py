"""
文本转为数字序列
"""
import os
from tqdm import tqdm
from dataset import tokenize


class WordToSequence:
    def __init__(self):
        """
        case 1: 句子可能出现长度不一致的问题，可以对短句子进行特殊字符填充
        case 2: 新出现的词语无法在词典中找到，可以用特殊字符代替
        """
        self.UNK_TAG = "UNK"  # 用特殊字符代替在词典中找不到的词，对应到数字为0
        self.PAD_TAG = "PAD"  # 填充短句子，对应到数字为1

        self.UNK = 0
        self.PAD = 1

        # 词语 - 索引的字典，初始时有两个标志记号
        self.word_to_idx = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }

        # 统计词频的字典
        self.count_dict = {}

        # 索引 - 单词的字典，用于将输出转化为句子
        self.idx_to_word = {}

    def count_word_frequence(self, sentence):
        """
        统计词频
        :param sentence [word1, word2, word3 ...]
        """

        for word in sentence:
            # 如果key没有出现在字典中，对应value就为1
            # 如果key在字典中出现过，就直接拿到对应的value，再加1
            self.count_dict[word] = self.count_dict.get(word, 0) + 1

    def build_vocab(self, min_count=5, max_count=None, num_word=None):
        """
        生成词语 - 索引的词典
        :param min_count: 词语在语料中的最小出现次数
        :param max_count: 词语在语料中的最大出现次数
        :param num_word: 一共保留多少个词语
        """

        # 删除词典中词频小于min_count，或者大于max_count的词语
        if min_count is not None:
            self.count_dict = {key: value for key, value in self.count_dict.items() if value > min_count}
        if max_count is not None:
            self.count_dict = {key: value for key, value in self.count_dict.items() if value < max_count}

        # 保留固定长度的词语，根据词频排序，选择前(长度)个词语及其索引
        if num_word is not None:
            # sorted(可迭代对象, 排序对象, 排序顺序)
            self.count_dict = dict(sorted(self.count_dict.items(), key=lambda x: x[-1], reverse=True)[:num_word])

        # 单词 - 索引的字典
        for word in self.count_dict:
            # 初始时，UNK和PAD占据两个位置。不断加入新元素时，word的索引不断增加
            self.word_to_idx[word] = len(self.word_to_idx)

        # 索引 - 单词的字典，用于将输出转化为句子
        self.idx_to_word = dict(zip(self.word_to_idx.values(), self.word_to_idx.keys()))  # 反转字典

    def word_to_sequence(self, sentence, max_len=None):
        """
        句子 -> 数字序列
        :param sentence [word1, word2, word3, ...]
        :param max_len 对句子进行填充或裁剪
        """
        if max_len is not None:
            if max_len > len(sentence):
                # 填充
                sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
            elif max_len < len(sentence):
                # 裁减
                sentence = sentence[:max_len]

        return [self.word_to_idx.get(word, self.UNK) for word in sentence]

    def sequence_to_word(self, indices):
        """
        数字序列 -> 句子序列输出
        :param indices: [1, 2, 3, ...]
        """
        return [self.idx_to_word.get(index) for index in indices]

    def __len__(self):
        return len(self.word_to_idx)


if __name__ == '__main__':
    from word_to_sequence import WordToSequence  # pickle的bug...
    import pickle

    ws = WordToSequence()

    path = './dataset/aclImdb/train'

    temp_data_path = [os.path.join(path, 'pos'), os.path.join(path, 'neg')]

    for data_path in temp_data_path:
        file_paths = [
            os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if file_name.endswith('.txt')
        ]

        for file_path in tqdm(file_paths):
            sentences = tokenize(open(file_path, encoding='utf-8').read())

            ws.count_word_frequence(sentences)

    ws.build_vocab(min_count=10, num_word=10000)

    if not os.path.exists('./model'):
        os.mkdir('./model')
    pickle.dump(ws, open("./model/ws.pkl", "wb"))

    print('word sequence length:', len(ws))
