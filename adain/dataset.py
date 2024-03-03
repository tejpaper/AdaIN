from .constants import TRANSFORMS, PATH2DATASET, DEVICE

import torch
import torch.nn as nn

from torch.utils.data import Dataset

import os
import random

from abc import ABC
from typing import Tuple

assert __name__ != '__main__', 'Module startup error.'


class DataSet(Dataset, ABC):
    def __init__(self, root: str = None, transform: nn.Module = None):
        if root is None:
            root = PATH2DATASET

        assert {'content', 'style'} <= set(os.listdir(root))

        content_dir = os.path.join(root, 'content')
        self.content_samples = [os.path.join(content_dir, path) for path in os.listdir(content_dir)]

        style_dir = os.path.join(root, 'style')
        self.style_samples = [os.path.join(style_dir, path) for path in os.listdir(style_dir)]

        assert len(self.content_samples) == len(self.style_samples)

        if transform is None:
            self.transform = TRANSFORMS
        else:
            self.transform = transform

        self.valid_content_list = None
        self.valid_style_list = None
        self.skip(index=0)

    def __len__(self) -> int:
        return len(self.valid_content_list)

    def skip(self, index: int):
        self.valid_content_list = self.content_samples.copy()[index:]
        self.valid_style_list = self.style_samples.copy()[index:]

    def shuffle(self, seed: int = 0):
        torch.manual_seed(seed)
        random.seed(seed)

        random.shuffle(self.content_samples)
        random.shuffle(self.style_samples)

        self.skip(index=0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path2content = self.valid_content_list[index]
        path2style = self.valid_style_list[index]

        content_sample = self.transform(path2content).to(DEVICE)
        style_sample = self.transform(path2style).to(DEVICE)

        return content_sample, style_sample
