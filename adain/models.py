from .constants import PATH2VGG
from .functions import requires_grad, adaptive_instance_normalization

import torch
import torch.nn as nn

from typing import List, Tuple, Union

assert __name__ != '__main__', 'Module startup error.'


class ReflectionConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(*args, **kwargs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.add_module(
            'relu1_1', nn.Sequential(
                nn.Conv2d(3, 3, (1, 1)),             # special conv layer
                ReflectionConv2d(3, 64, (3, 3)),                               # conv1_1
                nn.ReLU(),                                                           # relu1_1
            )
        )

        self.add_module(
            'relu2_1', nn.Sequential(
                ReflectionConv2d(64, 64, (3, 3)),                               # conv1_2
                nn.ReLU(),                                                            # relu1_2
                nn.MaxPool2d(2),                                                      # pooling
                ReflectionConv2d(64, 128, (3, 3)),                              # conv2_1
                nn.ReLU(),                                                            # relu2_1
            )
        )

        self.add_module(
            'relu3_1', nn.Sequential(
                ReflectionConv2d(128, 128, (3, 3)),                              # conv2_2
                nn.ReLU(),                                                             # relu2_2
                nn.MaxPool2d(2),                                                       # pooling
                ReflectionConv2d(128, 256, (3, 3)),                              # conv3_1
                nn.ReLU(),                                                             # relu3_1
            )
        )

        self.add_module(
            'relu4_1', nn.Sequential(
                ReflectionConv2d(256, 256, (3, 3)),                              # conv3_2
                nn.ReLU(),                                                             # relu3_2
                ReflectionConv2d(256, 256, (3, 3)),                              # conv3_3
                nn.ReLU(),                                                             # relu3_3
                ReflectionConv2d(256, 256, (3, 3)),                              # conv3_4
                nn.ReLU(),                                                             # relu3_4
                nn.MaxPool2d(2),                                                       # pooling
                ReflectionConv2d(256, 512, (3, 3)),                              # conv4_1
                nn.ReLU()                                                              # relu4_1
            )
        )

        self.__load_weights()

    def __load_weights(self):
        weights = torch.load(PATH2VGG)

        for new_key, key in zip([*self.state_dict()], [*weights]):
            weights[new_key] = weights.pop(key)

        self.load_state_dict(weights)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]

        for layer in self.children():
            features.append(layer(features[-1]))

        return features[1:]


class StyleTransferNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.extractor = FeatureExtractor()
        self.extractor.eval()
        requires_grad(self.extractor)

        self.adain = adaptive_instance_normalization

        self.decoder = nn.Sequential(
            ReflectionConv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            ReflectionConv2d(256, 256, (3, 3)),
            nn.ReLU(),
            ReflectionConv2d(256, 256, (3, 3)),
            nn.ReLU(),
            ReflectionConv2d(256, 256, (3, 3)),
            nn.ReLU(),
            ReflectionConv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            ReflectionConv2d(128, 128, (3, 3)),
            nn.ReLU(),
            ReflectionConv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            ReflectionConv2d(64, 64, (3, 3)),
            nn.ReLU(),
            ReflectionConv2d(64, 3, (3, 3)),
            nn.ReLU()
        )

    def train(self, mode: bool = True):
        self.training = mode

        requires_grad(self.decoder, mode)
        for layer in self.decoder.children():
            layer.train(mode)

        return self

    def mix(self, f_c: torch.Tensor, *style_features: Tuple[torch.Tensor, Union[int, float]]):
        assert len(style_features)

        style_impact = 0
        for _, multiplier in style_features:
            assert 0 <= multiplier <= 1
            style_impact += multiplier

        assert 0 <= style_impact <= 1
        t = (1 - style_impact) * f_c

        for f_s, multiplier in style_features:
            t = t + multiplier * self.adain(f_c, f_s)

        return self.decoder(t)

    def forward(self, content: torch.Tensor, style: torch.Tensor, alpha: Union[int, float] = 1) -> torch.Tensor:
        assert 0 <= alpha <= 1

        *_, f_c = self.extractor(content)
        *_, f_s = self.extractor(style)

        return self.mix(f_c, (f_s, alpha))