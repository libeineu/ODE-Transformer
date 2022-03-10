import torch
import torch.nn as nn
from fairseq.models.transformer import LayerNorm
import queue
import fairseq.utils as utils
import torch.nn.functional as F
import numpy as np


def CreateEfficientBlock(args, is_encoder, is_inner):
    history_type = args.encoder_inner_block_type if is_inner else args.encoder_outer_block_type if is_encoder else args.decoder_history_type

    if history_type is None:
        return None
    elif history_type == "residual":
        return ResidualLayerHistory(args, is_encoder, is_inner=is_inner)
    elif history_type == "dense":
        return DenseLayerHistory(args, is_encoder, is_inner=is_inner)
    elif history_type == "learnable_dense":
        return LearnableDenseLayerHistory(args, is_encoder, is_inner=is_inner)
    else:
        raise ValueError


class BaseLayerHistory(nn.Module):

    def __init__(self, args, is_encoder, is_inner):
        super(BaseLayerHistory, self).__init__()
        self.is_encoder = is_encoder
        self.normalize_before = args.encoder_normalize_before if is_encoder else args.decoder_normalize_before

        # the first layer (aka. embedding layer) does not have layer normalization
        assert args.encoder_layers % args.in_block_num == 0
        layers = args.in_block_num if is_inner else int(args.encoder_layers / args.in_block_num)
        dim = args.encoder_embed_dim
        self.layer_norms = nn.ModuleList(LayerNorm(dim) for _ in range(layers))

    def add(self, layer):
        raise NotImplemented

    def pop(self):
        raise NotImplemented

    def clean(self):
        raise NotImplemented

    def reset(self, x):
        raise NotImplemented


class ResidualLayerHistory(BaseLayerHistory):
    """
    x_n = x_{n-1} + y_{n-1}
    """

    def __init__(self, args, is_encoder, is_inner):
        super(ResidualLayerHistory, self).__init__(args, is_encoder, is_inner)
        self.count = 0
        self.x = None
        self.y = None

    def add(self, layer):
        if self.x is None:
            self.x = layer
            self.count += 1
            return
        self.count += 1
        if self.normalize_before:
            self.y = self.layer_norms[self.count - 2](layer)
        else:
            self.y = layer

    def pop(self):
        assert self.x is not None
        if self.y is None:
            return self.x
        ret = self.x + self.y
        if not self.normalize_before:
            ret = self.layer_norms[self.count - 2](ret)
        self.x = ret
        return ret

    def clean(self):
        self.x = None
        self.y = None
        self.count = 0


class DenseLayerHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, args, is_encoder, is_inner):
        super(DenseLayerHistory, self).__init__(args, is_encoder, is_inner)
        self.sum = None
        self.count = 0
        self.individuals = None  # store past individual value, used for windows_size > 0

        self.integration_type = getattr(args, 'encoder_integration_type', 'avg') if is_encoder else \
            getattr(args, 'decoder_integration_type', 'avg')


    def add(self, layer):
        self.count += 1

        # first layer
        if self.sum is None:
            self.sum = layer
            return

        # following layer
        if self.normalize_before:
            layer = self.layer_norms[self.count - 2](layer)

        self.sum = self.sum + layer

    def pop(self):
        assert self.sum is not None
        if self.integration_type == 'sum':
            ret = self.sum
        else:
            ret = self.sum / self.count
        if self.count == 1 or self.normalize_before:
            return ret
        return self.layer_norms[self.count - 2](ret)

    def clean(self):
        self.sum = None
        self.count = 0

    def reset(self, x):
        self.sum = x
        self.count = 1


class LearnableDenseLayerHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, args, is_encoder, is_inner):
        super(LearnableDenseLayerHistory, self).__init__(args, is_encoder, is_inner)
        self.sum = None
        self.count = 0
        self.layer_num = 1 + (args.encoder_layers if is_encoder else args.decoder_layers)
        self.weight = nn.Parameter(torch.Tensor(self.layer_num, self.layer_num).fill_(1.0).tril())
        self.weight.data = self.weight.data / self.weight.data.sum(1, keepdim=True)


    def add(self, layer):
        self.count += 1
        # first layer
        if self.sum is None:
            self.sum = layer
            self.layers.append(layer)
            return

        # following layer
        if self.normalize_before:
            layer = self.layer_norms[self.count - 2](layer)

        self.layers.append(layer)

    def pop(self):
        assert len(self.layers) > 0
        ret = (torch.stack(self.layers, 0) * self.weight[self.count - 1, : self.count].view(-1, 1, 1, 1)).sum(0)
        if self.count == 1 or self.normalize_before:
            return ret
        return self.layer_norms[self.count - 2](ret)

    def clean(self):
        self.sum = None
        self.count = 0
        self.layers = []

    def get_loss(self):
        return (0.5 * (self.weight.sum(1) - 1.0) ** 2).mean()
