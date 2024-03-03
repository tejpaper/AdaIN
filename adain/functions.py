from .constants import DEVICE

import torch
import torch.nn as nn

from torch.nn.functional import mse_loss

from typing import Tuple, List

assert __name__ != '__main__', 'Module startup error.'


def requires_grad(model: nn.Module, flag: bool = False):
    for param in model.parameters():
        param.requires_grad = flag


def calc_mean_std(x: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    n, c, *hw = x.size()
    x = x.view(n, c, -1)

    mean = x.mean(-1).view(n, c, 1, 1).expand(n, c, *hw)
    std = torch.sqrt(x.var(-1) + eps).view(n, c, 1, 1).expand(n, c, *hw)

    return mean, std


def mat_sqrt(x: torch.Tensor) -> torch.Tensor:
    u, d, v = torch.svd(x)
    return torch.mm(torch.mm(u, d.sqrt().diag()), v.t())


def adaptive_instance_normalization(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    content_mean, content_std = calc_mean_std(content)
    style_mean, style_std = calc_mean_std(style)

    c_norm = (content - content_mean) / content_std
    t_code = style_std * c_norm + style_mean

    return t_code


def color_fixation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inp_size = target.size()

    target = target.view(3, -1)
    source = source.view(3, -1)

    t_std, t_mean = torch.std_mean(target, dim=-1, keepdim=True)
    s_std, s_mean = torch.std_mean(source, dim=-1, keepdim=True)

    t_norm = (target - t_mean.expand_as(target)) / t_std.expand_as(target)
    s_norm = (source - s_mean.expand_as(source)) / s_std.expand_as(source)

    t_cov_eye = torch.mm(t_norm, t_norm.t()) + torch.eye(3, device=DEVICE)
    s_cov_eye = torch.mm(s_norm, s_norm.t()) + torch.eye(3, device=DEVICE)

    s_norm_transfer = torch.mm(mat_sqrt(t_cov_eye), torch.mm(torch.inverse(mat_sqrt(s_cov_eye)), s_norm))
    s_transfer = s_norm_transfer * t_std.expand_as(source) + t_mean.expand_as(source)

    return s_transfer.view(inp_size)


def content_loss(generated_features: torch.Tensor, t_code: torch.Tensor) -> torch.Tensor:
    assert not t_code.requires_grad
    return mse_loss(generated_features, t_code)


def style_loss(generated_features: List[torch.Tensor], style_features: List[torch.Tensor]) -> torch.Tensor:
    assert not any([f.requires_grad for f in style_features])
    assert len(generated_features) == len(style_features)

    loss = torch.tensor(0., requires_grad=True)

    for g_f, s_f in zip(generated_features, style_features):
        g_f_mean, g_f_std = calc_mean_std(g_f)
        s_f_mean, s_f_std = calc_mean_std(s_f)

        loss = loss + mse_loss(g_f_mean, s_f_mean) + mse_loss(g_f_std, s_f_std)

    return loss
