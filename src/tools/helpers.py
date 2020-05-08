import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
import os


from sklearn.model_selection import train_test_split
from copy import deepcopy

from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter

from .metrics import f1_per_frame, weak_label_metrics
from .settings import MODEL_PATH, MODEL_NAME, test_model_name, ENABLE_CLASS_COUNTING

from tqdm import tqdm

writer = SummaryWriter()


def get_samples_per_class(data_idx, dataset, enable_counting=False):
    """
    Get class distribution of the data

    :param enable_counting: indicate whether class distribution has to be counted. Disable for speed
    :param data_idx: indices of the divided data
    :param dataset: entire dataset
    :return: class distribution dictionary
    """
    if enable_counting is True:
        data_sampler = SubsetRandomSampler(data_idx)
        _dataloader = DataLoader(dataset, batch_size=1, sampler=data_sampler)



        class_dist = {k: 0 for k in dataset.classes}
        for idx, (_, _, weak_label) in enumerate(_dataloader):
            class_dist[weak_label[0]] += 1

        return class_dist
    else:
        return "Samples per class not counted"


def stratified_split(dataset,
                     mode="three_split",
                     test_split=0.2,
                     val_split=0.2,
                     batch_size=1,
                     num_workers=1,
                     seed=None):
    """
    Get stratified split of the given dataset
    :param dataset: dataset object
    :param mode: defines train/test or train/val/test split
    :param test_split: test split size relative to entire dataset
    :param val_split: validation split size relative to train dataset
    :param batch_size: Dataloader batch_size
    :param num_workers: Dataloader num_workers
    :param seed: Random seed for splitting
    :return: Train/Val/Test or Train/Test Dataloader objects
    """

    labels = dataset.targets
    avail_modes = ["three_split", "two_split"]

    if mode == avail_modes[0]:
        train_val_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=test_split, shuffle=True,
                                                   stratify=labels, random_state=seed)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_split, shuffle=True,
                                              stratify=labels[train_val_idx], random_state=seed)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler)

        data_dist = get_samples_per_class(np.arange(len(dataset)), dataset, ENABLE_CLASS_COUNTING)
        train_dist = get_samples_per_class(train_idx, dataset, ENABLE_CLASS_COUNTING)
        val_dist = get_samples_per_class(val_idx, dataset, ENABLE_CLASS_COUNTING)
        test_dist = get_samples_per_class(test_idx, dataset, ENABLE_CLASS_COUNTING)

        print("""
            Dataset details....
            -----------------
            Total samples: {}, {}
            Training: {}, {}
            Validation: {}, {}
            Testing: {}, {}
            ---------------""".format(len(dataset), data_dist,
                                      len(train_idx), train_dist,
                                      len(val_idx), val_dist,
                                      len(test_idx), test_dist))

        return train_loader, val_loader, test_loader
    elif mode == avail_modes[1]:
        train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=test_split, shuffle=True,
                                               stratify=labels, random_state=seed)

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler)

        data_dist = get_samples_per_class(np.arange(len(dataset)), dataset, ENABLE_CLASS_COUNTING)
        train_dist = get_samples_per_class(train_idx, dataset, ENABLE_CLASS_COUNTING)
        test_dist = get_samples_per_class(test_idx, dataset, ENABLE_CLASS_COUNTING)

        print("""
            Dataset details....
            -----------------
            Total samples: {}, {}
            Training: {}, {}
            Testing: {}, {}
            ---------------""".format(len(dataset), data_dist,
                                      len(train_idx), train_dist,
                                      len(test_idx), test_dist))
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
                  mode="train"):
    """
    Runs one epoch for either training, validation or testing.

    :param model: Unoptimized/Optimized model
    :param data: Data to be trained/validated/tested
    :param loss_function: loss/objective function to be used for optimising the model
    :param optimizer: Optimizer to be used
    :param device: Device chosen for the entire process (CUDA or CPU)
    :param epoch_num: Epoch number
    :param mode: Select between train/test/validation
    :return: model: Optimized/Unchanged model
    :return: f1_score: Returns f1_score (only in test mode)
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
        epoch_num = 1
        if optimizer is not None or loss_function is not None:
            raise ValueError("Optimizer/Loss function should be of {} for mode: {}".format(type(None), mode))
    else:
        raise ValueError("Selected mode ({}) not available. Choose from: {}".format(mode, avail_modes))

    running_loss = 0.0
    batch_size = data.batch_size
    best_val_loss = 1e8

    y_hat, y_true = [], []

    for idx, (sample, label, _) in enumerate(data):
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
            writer.add_scalar('{} Loss'.format(mode), running_loss, epoch_num * len(data) + idx)
        elif mode == "validation":
            writer.add_scalar('{} Loss'.format(mode), running_loss, epoch_num * len(data) + idx)
            if best_val_loss > running_loss:
                best_val_loss = running_loss

        running_loss = 0.0

    y_hat = torch.cat(y_hat, dim=0)
    y_true = torch.cat(y_true, dim=0)

    # start computing metrics from here
    f1_score = f1_per_frame(y_hat.sigmoid(), y_true).mean().item()

    if mode == "train":
        writer.add_scalar('F1-Score - {}'.format(mode), f1_score, epoch_num)
        return model
    elif mode == "validation":
        writer.add_scalar('F1-Score - {}'.format(mode), f1_score, epoch_num)
        return model, best_val_loss
    elif mode == "test":
        weak_label_metrics(y_hat, y_true)
        return model, f1_score


def train_model(model=None,
                train_data=None,
                val_data=None,
                loss_function=None,
                optimizer=None,
                device=torch.device("cpu"),
                epochs=50):
    """
    Trains a model.

    :param model: Unoptimized model
    :param train_data: Data to train
    :param val_data: Data to validate
    :param loss_function: loss/objective function to used to optimize the model
    :param optimizer: Optimizer to be used
    :param device: Device chosen to train on (CPU/CUDA)
    :param epochs: Number of epochs to train
    :return: best_model: Optimized/Trained model
    """

    print("------------------Training----------------")

    best_model = None
    best_epoch = -1
    best_val_loss = 1e8

    for epoch in tqdm(range(epochs), desc="Training the model"):
        # training the model
        model = model.train()

        model = run_one_epoch(model=model,
                              data=train_data,
                              loss_function=loss_function,
                              optimizer=optimizer,
                              device=device,
                              epoch_num=epoch,
                              mode="train")

        if val_data is not None:
            # check model against validation data
            model = model.eval()
            with torch.no_grad():
                model, val_loss_epoch = run_one_epoch(model=model,
                                                      data=val_data,
                                                      loss_function=loss_function,
                                                      optimizer=None,
                                                      device=device,
                                                      epoch_num=epoch,
                                                      mode="validation")

            if val_loss_epoch < best_val_loss:
                val_loss_epoch = best_val_loss
                best_epoch = epoch
                best_model = deepcopy(model.state_dict())

        else:
            warnings.warn("No Validation data specified....")
            best_model = deepcopy(model.state_dict())
            best_epoch = epoch

    print("Training Done.....")
    print("Best performing model on validation found at epoch: {}".format(best_epoch+1))

    print("Saving model......")
    torch.save(best_model, os.path.join(MODEL_PATH, MODEL_NAME))


def test_model(model=None,
               test_data=None,
               device=torch.device("cpu")):
    """
    Tests a given model

    :param model: Optimized/Trained model to test
    :param test_data: Test data
    :param device: Device to carry out the process on (CPU/CUDA)
    :return: None
    """
    print("-------------Testing-----------------")
    if model is None:
        raise ValueError("No model to test...")

    test_model_path = os.path.join(MODEL_PATH, test_model_name)
    if os.path.exists(test_model_path):
        model.load_state_dict(torch.load(test_model_path))
        model = model.eval()
        with torch.no_grad():
            _, test_f1_score = run_one_epoch(model=model,
                                             data=test_data,
                                             loss_function=None,
                                             optimizer=None,
                                             device=device,
                                             mode="test")

        print("F1 score on test set: {}".format(test_f1_score))
    else:
        raise FileNotFoundError("Selected model does not exist at: {}".format(test_model_path))



