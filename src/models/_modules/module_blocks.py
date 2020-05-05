from functools import reduce
from src.helpers import apply_layers
# from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, ReLU, \
    BatchNorm2d, Dropout2d, MaxPool2d, GRUCell, Linear


class CNN(Module):

    def __init__(self, cnn_channels=64, cnn_dropout=0):
        self.layer_1 = Sequential(
            Conv2d(
                in_channels=1, out_channels=cnn_channels,
                kernel_size=5, stride=1, padding=2),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, 5),
                      stride=(1, 5)),
            Dropout2d(cnn_dropout)
        )

        self.layer_2 = Sequential(
            Conv2d(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=5, stride=1,
                padding=2),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, 4),
                      stride=(1, 4)),
            Dropout2d(cnn_dropout)
        )

        self.layer_3 = Sequential(
            Conv2d(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=5, stride=1,
                padding=2),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, 2),
                      stride=(1, 2)),
            Dropout2d(cnn_dropout)
        )

        self.layers = [self.layer_1, self.layer_2, self.layer_3]

    def forward(self, x):

        return reduce(apply_layers, self.layers, x.unsqueeze(1))


class RNN(Module):

    def __init__(self, input_size, hidden_size=64):
        self.layer_1 = GRUCell(input_size=input_size, hidden_size=hidden_size, bias=True)

        self.layers = [self.layer_1]

    def forward(self, x):
        return reduce(apply_layers, self.layers, x)


class FNN(Module):

    def __init__(self, input_size, out_classes=2):
        self.layer_1 = Linear(in_features=input_size, out_features=out_classes, bias=True)

        self.layers = [self.layer_1]

    def forward(self, x):
        return reduce(apply_layers, self.layers, x)
