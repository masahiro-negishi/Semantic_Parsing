import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def train(epoch_num, seq2seq, train_iterator, optimizer, criterion, device):
    """train

    train model

    Args:
        epoch_num (int): epoch num
        seq2seq (nn.Module): model to train
        train_iterator (torchtext.data.Iterator): train iterator
        optimizer (nn.RMSprop): RMSprop
        criterion (nn.CrossEntropyLoss): cross entropy loss
        device (torch.device): cpu or cuda

    Returns:
        list of float: loss history
    """

    seq2seq.train()
    loss_hist = []

    for epoch in range(epoch_num):
        epoch_loss = 0
        for batch in train_iterator:
            text = batch.Text[0].to(device)
            form = batch.Form[0].to(device) # (max_len, batch_size)

            optimizer.zero_grad()

            outputs = seq2seq.train_forward(text, form) # (max_len, batch_size, form_v_size)

            loss = 0

            for i in range(1, outputs.shape[0]):
                loss += criterion(outputs[i], form[i])

            loss.backward()

            nn.utils.clip_grad_norm_(seq2seq.parameters(), 5)

            optimizer.step()

            epoch_loss += loss.item()

        print(epoch, epoch_loss / len(train_iterator))
        loss_hist.append(epoch_loss / len(train_iterator))
    return loss_hist


def plot_loss(loss_hist, title):
    """plot loss history

    plot loss history

    Args:
        loss_hist (list of float): loss history
        title (string): title of the graph
    """
    plt.plot(np.arange(len(loss_hist)), loss_hist)
    plt.title(title)