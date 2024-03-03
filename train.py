from adain import *
from adain.functions import content_loss, style_loss
from adain.utils import save_model, save_log

from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

from typing import Generator

LR = 1e-4
GAMMA = (1 / 9) ** (1 / 320)
BATCH_SIZE = 8
LAMBDA = 10
EPOCHS_NUM = 16
SAVE_FREQ = 500


def train(model: StyleTransferNetwork, path2ds: str = None, iter_num: int = 0) -> Generator[int, None, None]:
    training = model.training
    model.train()

    ds = DataSet(path2ds)

    optimizer = optim.Adam(model.decoder.parameters(), lr=LR * GAMMA ** (iter_num / 500))
    scheduler = StepLR(optimizer, step_size=500, gamma=GAMMA)

    for epoch in range((iter_num // 10000) + 1, EPOCHS_NUM + 1):
        content_losses = list()
        style_losses = list()

        ds.shuffle(seed=epoch)
        ds.skip(iter_num * BATCH_SIZE % ds.__len__())

        loader = tqdm(
            DataLoader(ds, BATCH_SIZE),
            desc=f'Epoch: {epoch}; samples',
            unit=' sample',
            unit_scale=BATCH_SIZE
        )

        for content, style in loader:
            optimizer.zero_grad()

            f_c = model.extractor(content)
            f_s = model.extractor(style)

            t = model.adain(f_c[-1], f_s[-1])
            g_samples = model.decoder(t)

            f_g = model.extractor(g_samples)

            content_losses.append(content_loss(f_g[-1], t))
            style_losses.append(style_loss(f_g, f_s))
            loss = content_losses[-1] + LAMBDA * style_losses[-1]

            loss.backward()
            optimizer.step()
            scheduler.step()

            iter_num += 1

            if iter_num % SAVE_FREQ == 0:
                yield iter_num

        mean_content_loss = sum(content_losses) / len(content_losses)
        mean_style_loss = sum(style_losses) / len(style_losses)

        print(f'Mean content loss: {mean_content_loss}',
              f'Mean style loss: {mean_style_loss}',
              f'Mean common loss: {mean_content_loss + LAMBDA * mean_style_loss}',
              sep='\n', end='\n\n')

    model.train(training)
    yield iter_num


def main():
    model = StyleTransferNetwork().to(DEVICE)
    for iter_num in train(model):
        save_model(model, iter_num)
        save_log(model, iter_num)


if __name__ == '__main__':
    main()
