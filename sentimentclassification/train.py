import config
from dataset import get_dataloader
import torch
from model import SentiClassModel
import pickle
import torch.nn.functional as F


def train(epochs, net, optim):
    net.train()

    for epoch in range(epochs):
        epoch += 1

        for batch_idx, (data, label) in enumerate(
                get_dataloader(train=True, batch_size=config.batch_size, shuffle=True)
        ):
            data = data.to(config.device)
            label = label.to(config.device)

            output = net(data).to(config.device)

            optim.zero_grad()
            loss = F.nll_loss(output, label)
            loss.backward()
            optim.step()

            if batch_idx % 10 == 0:
                print('epoch: {}, batch: {}, loss: {}'.format(
                    epoch, batch_idx, loss.item()
                ))

                torch.save(net.state_dict(), './model/model.pkl')
                torch.save(optim.state_dict(), './model/optimizer.pkl')


if __name__ == '__main__':
    ws = pickle.load(open('./model/ws.pkl', 'rb'))

    senti_class_model = SentiClassModel(len(ws)).to(config.device)

    optimizer = torch.optim.Adam(senti_class_model.parameters())

    train(1, senti_class_model, optimizer)
