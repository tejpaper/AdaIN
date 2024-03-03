from .constants import IMG2TENSOR, TENSOR2IMG, CHECKPOINTS_DIR, LOGS_DIR, CONTENT_IMG, STYLE_IMG
from .models import StyleTransferNetwork

import torch

import os
import matplotlib.pyplot as plt

assert __name__ != '__main__', 'Module startup error.'


def save_model(model: StyleTransferNetwork, iter_num: int) -> str:
    path = os.path.join(CHECKPOINTS_DIR, f'checkpoint-{iter_num}.pth')
    torch.save(model.state_dict(), path)

    return path


@torch.no_grad()
def save_log(model: StyleTransferNetwork, iter_num: int) -> str:
    content = IMG2TENSOR(CONTENT_IMG)
    style = IMG2TENSOR(STYLE_IMG)

    training = model.training
    model.eval()

    figure = plt.figure(figsize=(10, 2), dpi=340, frameon=False)
    plt.subplots_adjust(wspace=0)

    for i in range(5):
        alpha = i / 4

        tensor = model(content, style, alpha)
        image = TENSOR2IMG(tensor)

        axis = figure.add_subplot(1, 5, i + 1)
        plt.imshow(image)

        axis.set_xlabel(f'α = {alpha}')
        axis.set_xticks([])
        axis.set_yticks([])

    model.train(training)

    path = os.path.join(LOGS_DIR, f'{iter_num}.png')
    plt.savefig(path)
    plt.close()

    return path
