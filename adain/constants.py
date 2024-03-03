import torch

from torchvision import transforms

import os
from PIL import Image

assert __name__ != '__main__', 'Module startup error.'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = transforms.Lambda(lambda p: Image.open(p).convert('RGB'))

TRANSFORMS = transforms.Compose([
    loader,
    transforms.RandomCrop(256),
    transforms.ToTensor()
])

IMG2TENSOR = transforms.Compose([
    loader,
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.unsqueeze(0).to(DEVICE))
])

TENSOR2IMG = transforms.Compose([
    transforms.Lambda(lambda t: t.squeeze().clamp(0, 1)),
    transforms.ToPILImage()
])

PATH2DATASET = r'D:\App\Datasets\CS-COCO'

MODELS_DIR = 'models'
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')
PATH2VGG = os.path.join(MODELS_DIR, 'vgg19-norm.pth')
PATH2MODEL = os.path.join(MODELS_DIR, 'model.pth')

LOGS_DIR = 'logs'
CONTENT_IMG = os.path.join('extra', 'content.jpg')
STYLE_IMG = os.path.join('extra', 'style.jpg')
