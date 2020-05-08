import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa.display
import sys
import argparse

from tools.dataset_spec import PriusData
from tools.helpers import stratified_split, run_one_epoch, train_model, test_model
from tools.settings import DATA_DIR, SAMPLE_RATE, MEL_BANKS, \
    MIC_USED, DEVICE, EPOCHS, BATCH_SIZE, CLASS_TYPE, RANDOM_SEED, \
    CNN_CHANNELS, RNN_IN_SIZE, RNN_HH_SIZE, DROPOUT, \
    test_cnn_channels, test_rnn_in_size, test_rnn_hh_size, test_dropout

from models.baseline_crnn import CRNN

from torchvision import transforms
from torch import Tensor
from torch.nn import BCEWithLogitsLoss


def parse_args():
    class ExtractorArgsParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            sys.exit(2)

        def format_help(self):
            formatter = self._get_formatter()
            formatter.add_text(self.description)
            formatter.add_usage(self.usage, self._actions,
                                self._mutually_exclusive_groups)
            for action_group in self._action_groups:
                formatter.start_section(action_group.title)
                formatter.add_text(action_group.description)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()
            formatter.add_text(self.epilog)

            return formatter.format_help()

    usage = """
    Training a deep-learning based acoustic model to predict cars coming around corners
    """
    parser = ExtractorArgsParser(description='python main.py',
                                 usage=usage)
    parser.add_argument('--test',
                        help='Only test the desired model',
                        action='store_true')

    try:
        parsed = parser.parse_args()
    except:
        sys.exit(2)

    return parsed


if __name__ == '__main__':

    parsed = parse_args()

    if CLASS_TYPE == "coarse_class":
        out_classes = 2
    elif CLASS_TYPE == "fine_class":
        out_classes = 4
    else:
        raise ValueError("Error in class type selection!")

    # transforms as required
    my_transforms = transforms.Compose([
        # lambda x: x.astype(np.float32) / np.max(x), # rescale to -1 to 1
        # lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC
        lambda x: np.asfortranarray(x[:, MIC_USED]),
        lambda x: librosa.feature.melspectrogram(y=x, sr=SAMPLE_RATE, n_mels=MEL_BANKS, fmax=SAMPLE_RATE // 2),
        lambda x: Tensor(x)
    ])

    # initialize the dataset
    my_dataset = PriusData(DATA_DIR, transform=my_transforms, mode="static", class_type=CLASS_TYPE)

    # split the dataset
    train_data, val_data, test_data = stratified_split(my_dataset, mode="three_split",
                                                       batch_size=BATCH_SIZE, seed=RANDOM_SEED)

    if parsed.test is False:
        # setup the model, optimiser and the loss function
        my_model = CRNN(cnn_channels=CNN_CHANNELS,
                        rnn_in_size=RNN_IN_SIZE,
                        rnn_hh_size=RNN_HH_SIZE,
                        cnn_dropout=DROPOUT,
                        out_classes=out_classes).to(DEVICE)
        my_optim = torch.optim.Adam(my_model.parameters(), lr=0.001)
        my_loss = BCEWithLogitsLoss()

        # training the model
        train_model(model=my_model,
                    train_data=train_data,
                    val_data=val_data,
                    loss_function=my_loss,
                    optimizer=my_optim,
                    device=DEVICE,
                    epochs=EPOCHS)

        test_model_type = CRNN(cnn_channels=test_cnn_channels,
                               rnn_in_size=test_rnn_in_size,
                               rnn_hh_size=test_rnn_hh_size,
                               cnn_dropout=test_dropout,
                               out_classes=out_classes).to(DEVICE)
        test_model(model=test_model_type,
                   test_data=test_data,
                   device=DEVICE)

    else:
        test_model_type = CRNN(cnn_channels=test_cnn_channels, rnn_in_size=test_rnn_in_size,
                               rnn_hh_size=test_rnn_hh_size,
                               cnn_dropout=test_dropout, out_classes=out_classes).to(DEVICE)
        test_model(model=test_model_type,
                   test_data=test_data,
                   device=DEVICE)
