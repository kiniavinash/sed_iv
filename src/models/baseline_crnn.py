from torch import zeros, Tensor
from torch.nn import Module

from ._modules import CNN, GRU, FNN


class CRNN(Module):

    def __init__(self, cnn_channels=64, cnn_dropout=0, rnn_in_size=64, rnn_hh_size=64, out_classes=2):
        super().__init__()
        self.rnn_hh_size = rnn_hh_size
        self.out_classes = out_classes

        self.cnn = CNN(cnn_channels=cnn_channels, cnn_dropout=cnn_dropout)

        self.rnn = GRU(input_size=rnn_in_size, hidden_size=self.rnn_hh_size)

        self.classifier = FNN(input_size=self.rnn_hh_size, out_classes=self.out_classes)

    def forward(self, x):
        """Forward pass of the CRNN model.
        :param x: Input to the CRNN.
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        b_size, _, t_size = x.size()

        # convolve over the input spectrogram; reshape to stack the channels on top of each other
        cnn_output = self.cnn(x).reshape(b_size, -1, t_size)

        # initialize hidden states
        h = zeros(b_size, self.rnn_hh_size).to(x.device)

        # initialize output tensor
        outputs = zeros(b_size, self.out_classes, t_size).to(cnn_output.device)

        # for each time step, apply GRU (as per the reference paper) and the FNN
        for i, t_step in enumerate(cnn_output.permute(2, 1, 0)):
            h = self.rnn(t_step.T, h)
            outputs[:, :, i] = self.classifier(h)

        return outputs
