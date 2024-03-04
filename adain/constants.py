import os

from PIL import Image

import torch
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOADER = transforms.Lambda(lambda p: Image.open(p).convert('RGB'))

TRANSFORMS = transforms.Compose([
    LOADER,
    transforms.RandomCrop(256),
    transforms.ToTensor()
])

IMG2TENSOR = transforms.Compose([
    LOADER,
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.unsqueeze(0).to(DEVICE))
])

TENSOR2IMG = transforms.Compose([
    transforms.Lambda(lambda t: t.squeeze().clamp(0, 1)),
    transforms.ToPILImage()
])

PATH2DATASET = '/media/data/Datasets/CS-COCO'
""" Dataset structure is as follows:
.
├── content
│   ├── 0.jpg
│   ├── ...
│   └── 79999.jpg
└── style
    ├── 0.jpg
    ├── ...
    └── 79999.jpg
"""

MODELS_DIR = 'models'
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')
PATH2VGG = os.path.join(MODELS_DIR, 'vgg19-norm.pth')
PATH2MODEL = os.path.join(MODELS_DIR, 'model.pth')

IMAGES_DIR = 'images'
CONTENT_IMG = os.path.join(IMAGES_DIR, 'cornell university.jpg')
STYLE_IMG = os.path.join(IMAGES_DIR, 'woman with a hat.jpg')
LOGS_DIR = os.path.join('models', 'logs')
