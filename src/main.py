import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa.display

from tools.dataset_spec import PriusData
from tools.helpers import plot_spec, stratified_split
from models.baseline_crnn import CRNN

from torch.utils.data import DataLoader
from torchvision import transforms
from torch import Tensor

if __name__ == '__main__':

    data_dir = "../../data"
    sample_rate = 47998
    mel_banks = 128
    mic_used = 5  # mic number used
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms as required
    my_transforms = transforms.Compose([
        # lambda x: x.astype(np.float32) / np.max(x), # rescale to -1 to 1
        # lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC
        lambda x: np.asfortranarray(x[:, mic_used]),
        lambda x: librosa.feature.melspectrogram(y=x, sr=sample_rate, n_mels=mel_banks, fmax=sample_rate // 2),
        lambda x: Tensor(x)
    ])

    my_dataset = PriusData(data_dir, transform=my_transforms, mode="static")

    train_data, val_data, test_data = stratified_split(my_dataset, mode="three_split")

    my_model = CRNN().to(DEVICE)

    # go through the samples
    for idx, (sample, label) in enumerate(train_data):
        # debug only
        print("{}: DataShape: {}, Label: {}".format(idx, sample.shape, label))

        if DEVICE == torch.device("cuda"):
            sample = sample.cuda()

        output = my_model(sample)
        print("{}: OutputShape: {}". format(idx, output.shape))

        break




# set up dataloader for the entire dataset
# static_dataset = DataLoader(
#     my_dataset,
#     batch_size=1,
#     shuffle=True,
#     num_workers=1)
