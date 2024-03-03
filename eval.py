from adain import *
from adain.constants import IMG2TENSOR, TENSOR2IMG, PATH2MODEL, CONTENT_IMG, STYLE_IMG
from adain.functions import color_fixation

import torch
import torchvision.transforms.functional as func

import os

from PIL import Image

from typing import List, Union

MODEL = StyleTransferNetwork().to(DEVICE)
state_dict = torch.load(PATH2MODEL, map_location=DEVICE)
MODEL.load_state_dict(state_dict)
MODEL.eval()


class Style:
    def __init__(self, path: str, impact: Union[int, float] = 1, color_target: bool = False):
        self.path = path
        self.impact = impact
        self.color_target = color_target

    def __call__(self):
        return IMG2TENSOR(self.path)


@torch.no_grad()
def stylize(path2content: str,
            styles: List[Style],
            dir2save: str = None,
            save_color: bool = True,
            show_mode: bool = False,
            save_mode: bool = True
            ) -> Image.Image:

    if dir2save is None:
        dir2save = 'output'
    elif not save_mode:
        print('Image will not be saved!')

    filename = f'stylized-{os.path.basename(path2content)}'
    path = os.path.join(dir2save, filename)

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

        *_, f_s = MODEL.extractor(style_img)
        style_features.append((f_s, style.impact))

    *_, f_c = MODEL.extractor(content)

    tensor = MODEL.mix(f_c, *style_features)
    image = TENSOR2IMG(tensor)

    if show_mode:
        image.show()
    if save_mode:
        image.save(path, 'PNG')

    return image


def main():
    stylize(  # example
        CONTENT_IMG,
        [
            Style(STYLE_IMG)
        ],
        save_color=False,
        show_mode=True,
        save_mode=False
    )


if __name__ == '__main__':
    main()
