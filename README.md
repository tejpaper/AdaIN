# Adaptive Instance Normalization

## Description

Unofficial implementation of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) written in PyTorch together with code for training the network and its simple usage.

![Some qualitative examples](extra/examples.jpg "adfs")

## Usage

You only need to be familiar with two files to use this tool:

1. `inference.ipynb` contains some simple examples of style transfer, it is the basic file for image processing
2. `train.ipynb` trains the model from scratch. Use it if you want to customize the training process for yourself

Adaptations of these notebooks for Google Colab are contained in the `colab` folder.

## Installation

You need to have Python 3.11+. And run the following commands:

```bash
git clone https://github.com/tejpaper/AdaIN.git
cd AdaIN
pip install -r requirements.txt
```

## License

MIT