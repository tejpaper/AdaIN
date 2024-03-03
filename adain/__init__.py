if __name__ != '__main__':
    from .constants import DEVICE
    from .dataset import DataSet
    from .models import StyleTransferNetwork
else:
    raise ImportError('Package startup error.')
