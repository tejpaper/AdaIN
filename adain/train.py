from adain.constants import *
from adain.dataset import DataSet
from adain.functions import calc_content_loss, calc_style_loss
from adain.models import StyleTransferNetwork

import os
import typing
from dataclasses import dataclass

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    path2ds: str = PATH2DATASET
    checkpoints_dir: str = CHECKPOINTS_DIR
    content_img: str = CONTENT_IMG
    style_img: str = STYLE_IMG
    logs_dir: str = LOGS_DIR

    lr: float = 1e-4
    gamma: float = (1 / 9) ** (1 / 320)
    batch_size: int = 8
    style_weight: float = 10
    epoch_num: int = 16
    save_freq: int = 500


def save_model(model: StyleTransferNetwork, config: TrainingConfig, iter_num: int) -> str:
    path = os.path.join(config.checkpoints_dir, f'checkpoint-{iter_num}.pth')
    torch.save(model.state_dict(), path)

    return path


@torch.no_grad()
def save_log(model: StyleTransferNetwork, config: TrainingConfig, iter_num: int) -> str:
    content = IMG2TENSOR(config.content_img)
    style = IMG2TENSOR(config.style_img)

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

    path = os.path.join(config.logs_dir, f'{iter_num}.png')
    plt.savefig(path)
    plt.close()

    return path


def train(model: StyleTransferNetwork,
          config: TrainingConfig,
          iter_num: int = 0,
          ) -> typing.Generator[int, None, None]:

    training = model.training
    model.train()

    ds = DataSet(config.path2ds)

    optimizer = optim.Adam(model.decoder.parameters(), lr=config.lr * config.gamma ** (iter_num / 500))
    scheduler = StepLR(optimizer, step_size=500, gamma=config.gamma)

    for epoch in range((iter_num // 10000) + 1, config.epoch_num + 1):
        content_losses = list()
        style_losses = list()

        ds.shuffle(seed=epoch)
        ds.skip(iter_num * config.batch_size % ds.__len__())

        loader = tqdm(
            DataLoader(ds, config.batch_size, pin_memory=next(model.parameters()).is_cuda),
            desc=f'Epoch: {epoch}; samples',
            unit=' sample',
            unit_scale=config.batch_size,
        )

        for content, style in loader:
            content = content.to(DEVICE)
            style = style.to(DEVICE)

            optimizer.zero_grad()

            f_c = model.extractor(content)
            f_s = model.extractor(style)

            t = model.adain(f_c[-1], f_s[-1])
            g_samples = model.decoder(t)

            f_g = model.extractor(g_samples)

            content_loss = calc_content_loss(f_g[-1], t)
            style_loss = calc_style_loss(f_g, f_s)
            loss = content_loss + config.style_weight * style_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            content_losses.append(content_loss.item())
            style_losses.append(style_loss.item())

            iter_num += 1
            if iter_num % config.save_freq == 0:
                yield iter_num

        mean_content_loss = sum(content_losses) / len(content_losses)
        mean_style_loss = sum(style_losses) / len(style_losses)

        print(f'Mean content loss: {mean_content_loss}',
              f'Mean style loss: {mean_style_loss}',
              f'Mean common loss: {mean_content_loss + config.style_weight * mean_style_loss}',
              sep='\n', end='\n\n')

    model.train(training)
    yield iter_num
