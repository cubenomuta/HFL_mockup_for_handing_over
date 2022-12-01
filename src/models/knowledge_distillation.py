import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_model import Net


def mutual_train(
    client_net: Net,
    meme_net: Net,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    alpha: float,
    beta: float,
    device: torch.device,
):
    client_optimizer = torch.optim.SGD(client_net.parameters(), lr=lr)
    meme_optimizer = torch.optim.SGD(meme_net.parameters(), lr=lr)
    client_net.to(device)
    meme_net.to(device)
    for _ in range(epochs):
        meme_net.eval()
        client_net.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                meme_outputs = meme_net(images)
            client_outputs = client_net(images)
            client_optimizer.zero_grad()
            loss = loss_kd(client_outputs, labels, meme_outputs, alpha)
            loss.backward()
            client_optimizer.step()
        client_net.eval()
        meme_net.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                client_outputs = client_net(images)
            meme_outputs = meme_net(images)
            meme_optimizer.zero_grad()
            loss = loss_kd(meme_outputs, labels, client_outputs, beta)
            loss.backward()
            meme_optimizer.step()


def loss_kd(outputs, labels, teacher_outputs, alpha):
    loss = alpha * nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(outputs, dim=1), F.softmax(teacher_outputs, dim=1)
    ) + (1 - alpha) * F.cross_entropy(outputs, labels)
    return loss
