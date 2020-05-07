import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter

from .metrics import f1_per_frame

writer = SummaryWriter()


def stratified_split(dataset, mode="three_split", test_split=0.2, val_split=0.2, batch_size=1, num_workers=1):
    """
    Get stratified split of the given dataset
    :param dataset: dataset object
    :param mode: defines train/test or train/val/test split
    :param test_split: test split size relative to entire dataset
    :param val_split: validation split size relative to train dataset
    :param batch_size: Dataloader batch_size
    :param num_workers: Dataloader num_workers
    :return: Train/Val/Test or Train/Test Dataloader objects
    """
    #TODO: **kwargs
    labels      = dataset.targets
    avail_modes = ["three_split", "two_split"]

    if mode == avail_modes[0]:
        train_val_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=test_split, shuffle=True,
                                                   stratify=labels)
        train_idx, val_idx = train_test_split(np.arange(len(train_val_idx)), test_size=val_split, shuffle=True,
                                              stratify=labels[train_val_idx])

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler   = SubsetRandomSampler(val_idx)
        test_sampler  = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
        val_loader   = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler)
        test_loader  = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler)

        print("""
            Dataset details....
            -----------------
            Total samples: {}
            Training: {}
            Validation: {}
            Testing: {}
            ---------------""".format(len(dataset), len(train_loader)*batch_size,
                                      len(val_loader)*batch_size, len(test_loader)*batch_size))

        return train_loader, val_loader, test_loader

    elif mode == avail_modes[1]:
        train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=test_split, shuffle=True,
                                                   stratify=labels)

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler  = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
        test_loader  = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler)

        print("""
            Dataset details....
            -----------------
            Total samples: {}
            Training: {}
            Testing: {}
            ---------------""".format(len(dataset), len(train_loader)*batch_size,
                                      len(test_loader)*batch_size))

        return train_loader, test_loader

    else:
        raise ValueError("Invalid mode ({}) selected. Select among: {}".format(mode, avail_modes))


def plot_spec(sample_tensor, sample_rate):
    """
    Plot spectrogram of each sample
    :param sample_tensor: spectrogram in torch.Tensor
    :param sample_rate: audio sample rate
    :return: None
    """
    plt.figure(figsize=(10, 4))
    s_db = librosa.power_to_db(sample_tensor.numpy()[0, :, :], ref=np.max)
    librosa.display.specshow(s_db, x_axis='time', y_axis='mel', sr=sample_rate, fmax=20000)
    plt.colorbar(format='%+2.0f dB')
    plt.show()


def apply_layers(layer_input, layer):
    """
    Assist in forward pass of layers
    :param layer_input: Input to the layer
    :param layer: Layer as defined by the model
    :return: Output of the Layer
    """
    return layer(layer_input)


def run_one_epoch(model=None,
                  data=None,
                  loss_function=None,
                  optimizer=None,
                  device=torch.device("cpu"),
                  epoch_num=1,
                  mode="train",
                  batch_size=1):
    """
    Runs one epoch for either training, validation or testing.
    :param model: Unoptimized/Optimized model
    :param data: Data to be trained/validated/tested
    :param loss_function: loss/objective function to be used for optimising the model
    :param optimizer: Optimizer to be used
    :param device: Device chosen for the entire process (CUDA or CPU)
    :param epoch_num: Epoch number
    :param mode: Select between train/test/validation
    :param batch_size: Batch size
    :return: model: Optimized/Unchanged model
    """

    if model is None or data is None:
        raise TypeError("Model/Data can be of type: {}".format(type(None)))

    avail_modes = ["train", "validation", "test"]

    if mode == "train":
        if optimizer is None or loss_function is None:
            raise ValueError("Optimizer/Loss function cannot be {} for mode: {}".format(type(None), mode))

    elif mode == "validation":
        if optimizer is not None:
            raise ValueError("Optimizer should be of {} for mode: {}".format(type(None), mode))

        if loss_function is None:
            raise ValueError("Mode: {} needs a loss function".format(mode))

    elif mode == "test":
        if optimizer is not None or loss_function is not None:
            raise ValueError("Optimizer/Loss function should be of {} for mode: {}".format(type(None), mode))

    else:
        raise ValueError("Selected mode ({}) not available. Choose from: {}".format(mode, avail_modes))

    running_loss = 0.0

    y_hat, y_true = [], []

    for idx, (sample, label) in enumerate(data):
        # debug only
        # print("{}: DataShape: {}, LabelShape: {}, {}".format(idx, sample.shape, label))

        # setting all gradients to zero, apparently useful,
        # see: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        if optimizer is not None:
            optimizer.zero_grad()

        if device == torch.device("cuda"):
            sample = sample.cuda()
            label = label.cuda()

        label_hat = model(sample)

        # compute loss and backpropagate
        if loss_function is not None:
            loss = loss_function(label_hat, label)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            # TODO: not sure if I have to clip the gradients here.
            running_loss += loss.item() * batch_size

        y_true.append(label.cpu())
        y_hat.append(label_hat.cpu())

        if mode == "train":
            writer.add_scalar('Training Loss', running_loss, epoch_num * len(data) + idx)

        elif mode == "validation":
            writer.add_scalar('Validation Loss', running_loss, epoch_num * len(data) + idx)

        running_loss = 0.0

    y_hat  = torch.cat(y_hat, dim=0)
    y_true = torch.cat(y_true, dim=0)

    f1_score = f1_per_frame(y_hat.sigmoid(), y_true).mean().item()

    if mode == "train":
        writer.add_scalar('Training Mean F1-Score', f1_score, epoch_num)

    elif mode == "validation":
        writer.add_scalar('Validation Mean F1-score', f1_score, epoch_num)

    return model
