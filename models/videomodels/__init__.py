###
# Author: Kai Li
# Date: 2021-06-21 12:04:03
# LastEditors: Kai Li
# LastEditTime: 2021-07-08 11:34:08
###

from .resnet import ResNet, BasicBlock
from .shufflenetv2 import ShuffleNetV2
from .frcnn_videomodel import FRCNNVideoModel, update_frcnn_parameter

__all__ = [
    "ResNet",
    "BasicBlock",
    "ShuffleNetV2",
    "FRCNNVideoModel",
    "update_frcnn_parameter",
]


def register_model(custom_model):
    """Register a custom model, gettable with `models.get`.

    Args:
        custom_model: Custom model to register.

    """
    if custom_model.__name__ in globals().keys() or custom_model.__name__.lower() in globals().keys():
        raise ValueError(f"Model {custom_model.__name__} already exists. Choose another name.")
    globals().update({custom_model.__name__: custom_model})


def get(identifier):
    """Returns an model class from a string (case-insensitive).

    Args:
        identifier (str): the model name.

    Returns:
        :class:`torch.nn.Module`
    """
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret model name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret model name : {str(identifier)}")
