import librosa
import numpy as np

import matplotlib.pyplot as plt
import librosa.display

from dataset_spec import PriusData
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import Tensor


# plot spectrogram of each sample
def plot_spec(sample_tensor):
    plt.figure(figsize=(10, 4))
    s_db = librosa.power_to_db(sample_tensor.numpy()[0, :, :], ref=np.max)
    librosa.display.specshow(s_db, x_axis='time', y_axis='mel', sr=sample_rate, fmax=20000)
    plt.colorbar(format='%+2.0f dB')
    plt.show()


if __name__ == '__main__':

    data_dir = "../../data"
    sample_rate = 47998
    mel_banks = 128
    mic_used = 5          #mic number used

    # transforms as required
    my_transforms = transforms.Compose([
        # lambda x: x.astype(np.float32) / np.max(x), # rescale to -1 to 1
        # lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC
        lambda x: np.asfortranarray(x[:, mic_used]),
        lambda x: librosa.feature.melspectrogram(y=x, sr=sample_rate, n_mels=mel_banks, fmax=sample_rate // 2),
        lambda x: Tensor(x)
    ])

    # set up dataloader
    static_dataset = DataLoader(
        PriusData(data_dir, transform=my_transforms, mode="static"),
        batch_size=1,
        shuffle=True,
        num_workers=0)

    # go through the samples
    for idx, (sample, label) in enumerate(static_dataset):

        # debug only
        # print("{}: DataShape: {}, Label: {}".format(idx, sample.shape, label))
        # plot_spec(sample)

        break
