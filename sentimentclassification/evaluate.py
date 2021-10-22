from dataset import get_dataloader
import config
import numpy as np
import torch
from model import SentiClassModel
import os
import pickle
import torch.nn.functional as F


def evaluate(net):
    net.eval()

    loss_list = []
    acc_list = []

    for batch_idx, (data, label) in enumerate(
            get_dataloader(train=False, batch_size=config.batch_size, shuffle=True)
    ):
        data = data.to(config.device)
        label = label.to(config.device)

        with torch.no_grad():
            output = net(data)

            cur_loss = F.nll_loss(output, label)
            loss_list.append(cur_loss.cpu().item())

            """
            output: 一个[64, 2]的矩阵，表示共有batch个数据，每一行为一个softmax层之后的概率预测结果
            output.max(dim=-1): 对最后一维(此处为预测结果那一维)取每一行的最大值，并增加indices记录最大值的索引
            output.max(dim=-1)[-1]: 取出indices
            
            此处的max()函数实际上是torch.max()，只是因为output是一个torch张量，所以可以直接调用max方法
            """
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(label).float().mean()
            acc_list.append(cur_acc.cpu().item())

            if batch_idx % 10 == 0:
                print('batch: {}, loss: {}, acc: {}'.format(
                    batch_idx, cur_loss.item(), cur_acc.item()
                ))

    print('total loss: {}, total acc: {}'.format(
        np.mean(loss_list), np.mean(acc_list)
    ))


if __name__ == '__main__':
    ws = pickle.load(open('./model/ws.pkl', 'rb'))

    senti_class_model = SentiClassModel(len(ws)).to(config.device)

    optimizer = torch.optim.Adam(senti_class_model.parameters())

    if os.path.exists('./model/model.pkl'):
        senti_class_model.load_state_dict(torch.load('./model/model.pkl'))
        optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))

    evaluate(senti_class_model)
