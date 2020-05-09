import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa.display

from tools.dataset_spec import PriusData
from tools.helpers import stratified_split, run_one_epoch, train_model, test_model, category_split
from tools.settings import DATA_DIR, SAMPLE_RATE, MEL_BANKS, \
    MIC_USED, DEVICE, EPOCHS, BATCH_SIZE, CLASS_TYPE, RANDOM_SEED, \
    CNN_CHANNELS, RNN_IN_SIZE, RNN_HH_SIZE, DROPOUT, LR, OUT_CLASSES, \
    test_cnn_channels, test_rnn_in_size, test_rnn_hh_size, test_dropout

from models.baseline_crnn import CRNN

from torchvision import transforms
from torch import Tensor
from torch.nn import BCEWithLogitsLoss


def get_transform():

    # transforms as required
    my_transforms = transforms.Compose([
        # lambda x: x.astype(np.float32) / np.max(x), # rescale to -1 to 1
        # lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC
        lambda x: np.asfortranarray(x[:, MIC_USED]),
        lambda x: librosa.feature.melspectrogram(y=x, sr=SAMPLE_RATE, n_mels=MEL_BANKS, fmax=SAMPLE_RATE // 2),
        lambda x: Tensor(x)
    ])

    return my_transforms


def run_train_test(parsed, train_data=None, test_data=None, val_data=None):

    # (train and test)/test the model
    if parsed.test is False:
        # setup the model, optimiser and the loss function
        my_model = CRNN(cnn_channels=CNN_CHANNELS,
                        rnn_in_size=RNN_IN_SIZE,
                        rnn_hh_size=RNN_HH_SIZE,
                        cnn_dropout=DROPOUT,
                        out_classes=OUT_CLASSES).to(DEVICE)
        my_optim = torch.optim.Adam(my_model.parameters(), lr=LR)
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
                               out_classes=OUT_CLASSES).to(DEVICE)
        test_model(model=test_model_type,
                   test_data=test_data,
                   device=DEVICE)

    else:
        test_model_type = CRNN(cnn_channels=test_cnn_channels, rnn_in_size=test_rnn_in_size,
                               rnn_hh_size=test_rnn_hh_size,
                               cnn_dropout=test_dropout, out_classes=OUT_CLASSES).to(DEVICE)
        test_model(model=test_model_type,
                   test_data=test_data,
                   device=DEVICE)


def only_static_mixed_loc(parsed):

    print("==========Only Static samples - no location specific split==============")

    # set up the experiment
    my_transforms = get_transform()

    # initialize the dataset
    my_dataset = PriusData(DATA_DIR, transform=my_transforms, mode="static", class_type=CLASS_TYPE)

    # split the dataset
    train_data, val_data, test_data = stratified_split(my_dataset, mode="three_split",
                                                       batch_size=BATCH_SIZE, seed=RANDOM_SEED)

    run_train_test(parsed,
                   train_data=train_data,
                   test_data=test_data,
                   val_data=val_data)


def only_static_location_split(parsed):

    print("==========Only Static samples - location specific split==============")

    # set up the experiment
    my_transforms = get_transform()

    train_data, val_data, test_data = category_split(train_cat="static", test_cat="driving",
                                                     batch_size=BATCH_SIZE, seed=RANDOM_SEED,
                                                     transform=my_transforms)
    run_train_test(parsed,
                   train_data=train_data,
                   test_data=test_data,
                   val_data=val_data)
