import librosa
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler, DataLoader


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

        return train_loader, val_loader, test_loader

    elif mode == avail_modes[1]:
        train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=test_split, shuffle=True,
                                                   stratify=labels)

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler  = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
        test_loader  = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler)

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
