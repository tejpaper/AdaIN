from adain.constants import IMG2TENSOR, TENSOR2IMG
from adain.functions import color_fixation
from adain.models import StyleTransferNetwork

import os

import torch
import torchvision.transforms.functional as func
from PIL import Image


class Style:
    def __init__(self, path: str, weight: int | float = 1, color_target: bool = False) -> None:
        self.path = path
        self.weight = weight
        self.color_target = color_target

    def __call__(self) -> torch.Tensor:
        return IMG2TENSOR(self.path)


@torch.no_grad()
def stylize(model: StyleTransferNetwork, *,
            path2content: str,
            styles: list[Style],
            dir2save: str = None,
            save_color: bool = True,
            show_mode: bool = False,
            ) -> Image.Image:

    content = IMG2TENSOR(path2content)

    styles = sorted(styles, key=lambda s: not s.color_target)
    assert (len(styles) == 1) or not styles[1].color_target
    style_features = list()

    for style in styles:
        style_img = style()

        if content.size() != style_img.size():
            style_img = func.resize(style_img, content.size()[-2:])

        if save_color:
            if style.color_target:
                content = color_fixation(content, style_img)
            else:
                style_img = color_fixation(style_img, content)

        *_, f_s = model.extractor(style_img)
        style_features.append((f_s, style.weight))

    *_, f_c = model.extractor(content)

    tensor = model.mix(f_c, *style_features)
    image = TENSOR2IMG(tensor)

    if show_mode:
        image.show()

    if dir2save is not None:
        filename = f'stylized-{os.path.basename(path2content)}'
        path = os.path.join(dir2save, filename)
        image.save(path, 'PNG')

    return image
