import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa.display

from tools.dataset_spec import PriusData
from tools.helpers import stratified_split, run_one_epoch, train_model, \
    test_model, category_split, combo_split, bag_split
from tools.settings import DATA_DIR, SAMPLE_RATE, MEL_BANKS, \
    MIC_USED, DEVICE, EPOCHS, BATCH_SIZE, CLASS_TYPE, RANDOM_SEED, \
    CNN_CHANNELS, RNN_IN_SIZE, RNN_HH_SIZE, DROPOUT, LR, OUT_CLASSES, \
    test_cnn_channels, test_rnn_in_size, test_rnn_hh_size, test_dropout, \
    OPTIM_FUNC, N_FFT, OVERLAP, WINDOW

from models.baseline_crnn import CRNN

from torchvision import transforms
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter


def get_transform():
    # transforms as required
    my_transforms = transforms.Compose([
        # lambda x: x.astype(np.float32) / np.max(x), # rescale to -1 to 1
        # lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC
        lambda x: np.asfortranarray(x[:, MIC_USED]),
        lambda x: librosa.feature.melspectrogram(y=x, sr=SAMPLE_RATE, n_mels=MEL_BANKS, fmax=SAMPLE_RATE // 2,
                                                 n_fft=N_FFT, hop_length=int(N_FFT*OVERLAP), window=WINDOW),
        # lambda x: librosa.power_to_db(x, ref=np.max),
        lambda x: Tensor(x)
    ])

    return my_transforms


def run_train_test(parsed,
                   train_data=None,
                   test_data=None,
                   val_data=None,
                   writer=None,
                   seed=None,
                   verbose=True,
                   suffix=""):
    # (train and test)/test the model
    if parsed.test is False:
        # setup the model, optimiser and the loss function
        torch.manual_seed(RANDOM_SEED)
        my_model = CRNN(cnn_channels=CNN_CHANNELS,
                        rnn_in_size=RNN_IN_SIZE,
                        rnn_hh_size=RNN_HH_SIZE,
                        cnn_dropout=DROPOUT,
                        out_classes=OUT_CLASSES).to(DEVICE)

        if OPTIM_FUNC == "Adam":
            my_optim = torch.optim.Adam(my_model.parameters(), lr=LR)
        else:
            my_optim = torch.optim.SGD(my_model.parameters(), lr=LR)

        my_loss = BCEWithLogitsLoss()

        # training the model
        train_model(model=my_model,
                    train_data=train_data,
                    val_data=val_data,
                    loss_function=my_loss,
                    optimizer=my_optim,
                    device=DEVICE,
                    epochs=EPOCHS,
                    writer=writer,
                    seed=seed,
                    suffix=suffix)

        test_model_type = CRNN(cnn_channels=test_cnn_channels,
                               rnn_in_size=test_rnn_in_size,
                               rnn_hh_size=test_rnn_hh_size,
                               cnn_dropout=test_dropout,
                               out_classes=OUT_CLASSES).to(DEVICE)
        f1_score, conf_mat, accuracy = test_model(model=test_model_type,
                                                  test_data=test_data,
                                                  device=DEVICE,
                                                  verbose=verbose,
                                                  seed=seed,
                                                  suffix=suffix)
    else:
        # TODO: might not work as intended, to be checked later
        test_model_type = CRNN(cnn_channels=test_cnn_channels, rnn_in_size=test_rnn_in_size,
                               rnn_hh_size=test_rnn_hh_size,
                               cnn_dropout=test_dropout, out_classes=OUT_CLASSES).to(DEVICE)
        f1_score, conf_mat, accuracy = test_model(model=test_model_type,
                                                  test_data=test_data,
                                                  device=DEVICE,
                                                  verbose=verbose)

    return f1_score, conf_mat, accuracy


def only_static_mixed_loc(parsed):
    print("==========Only Static samples - no location specific split==============")

    # set up the experiment
    my_transforms = get_transform()

    # initialize the dataset
    my_dataset = PriusData(DATA_DIR, transform=my_transforms, mode="static", class_type=CLASS_TYPE)

    # split the dataset
    train_data, val_data, test_data = stratified_split(my_dataset, mode="three_split",
                                                       batch_size=BATCH_SIZE, seed=RANDOM_SEED)

    test_f1_score, test_conf_mat, test_accuracy = run_train_test(parsed,
                                                                 train_data=train_data,
                                                                 test_data=test_data,
                                                                 val_data=val_data)

    return test_f1_score, test_conf_mat, test_accuracy


def custom_train_test_split(parsed, train_set="static", test_set=None, seed=None):
    print("========================================================")
    print("Train Set: {}, Test set: {}".format(train_set, test_set))

    # set up the experiment
    my_transforms = get_transform()

    if type(train_set) is not tuple and type(test_set) is not tuple:
        if test_set is not None:
            suffix = "_train_" + train_set + "_test_" + test_set
        else:
            suffix = "_train_test_" + train_set

        writer = SummaryWriter(comment=suffix)

        train_data, val_data, test_data = category_split(train_cat=train_set, test_cat=test_set,
                                                         batch_size=BATCH_SIZE, seed=seed,
                                                         transform=my_transforms, verbose=False)
        test_f1_score, test_conf_mat, test_accuracy = run_train_test(parsed,
                                                                     train_data=train_data,
                                                                     test_data=test_data,
                                                                     val_data=val_data,
                                                                     writer=writer,
                                                                     verbose=False,
                                                                     seed=seed,
                                                                     suffix=suffix)
    else:
        if type(train_set) is tuple and type(test_set) is not tuple:
            suffix = "_train_" + "_".join(train_set) + "_test_" + test_set
        elif type(test_set) is tuple and type(train_set) is not tuple:
            suffix = "_train_" + train_set + "_test_" + "_".join(test_set)
        else:
            suffix = "_train_" + "_".join(train_set) + "_test_" + "_".join(test_set)

        writer = SummaryWriter(comment=suffix)

        train_data, val_data, test_data = combo_split(train_cat=train_set, test_cat=test_set,
                                                      batch_size=BATCH_SIZE, seed=seed,
                                                      transform=my_transforms, verbose=False)

        test_f1_score, test_conf_mat, test_accuracy = run_train_test(parsed,
                                                                     train_data=train_data,
                                                                     test_data=test_data,
                                                                     val_data=val_data,
                                                                     writer=writer,
                                                                     seed=seed,
                                                                     verbose=False,
                                                                     suffix=suffix)
    return test_f1_score, test_conf_mat, test_accuracy


def window_length_exp(parsed, data_set="static", seed=None):
    # change DATA_DIR in the settings file as required
    print("========================================================")
    print("Data Set: {}".format(data_set))
    # set up the experiment
    my_transforms = get_transform()

    suffix = "bagSplit_train_test_" + data_set

    writer = SummaryWriter(comment=suffix)

    train_data, val_data, test_data = bag_split(data_cat=data_set,
                                                batch_size=BATCH_SIZE,
                                                seed=seed,
                                                transform=my_transforms,
                                                verbose=False)

    test_f1_score, test_conf_mat, test_accuracy = run_train_test(parsed,
                                                                 train_data=train_data,
                                                                 test_data=test_data,
                                                                 val_data=val_data,
                                                                 writer=writer,
                                                                 seed=seed,
                                                                 verbose=True,
                                                                 suffix=suffix)
    return test_f1_score, test_conf_mat, test_accuracy
