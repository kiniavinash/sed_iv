import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa.display

from tools.dataset_spec import PriusData
from tools.helpers import stratified_split, run_one_epoch

from models.baseline_crnn import CRNN

from torch.utils.data import DataLoader
from torchvision import transforms
from torch import Tensor, no_grad
from torch.nn import BCEWithLogitsLoss
# from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# writer = SummaryWriter()

if __name__ == '__main__':

    data_dir = "../../data"
    sample_rate = 47998
    mel_banks = 128
    mic_used = 5  # mic number used
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    batch_size = 16

    # transforms as required
    my_transforms = transforms.Compose([
        # lambda x: x.astype(np.float32) / np.max(x), # rescale to -1 to 1
        # lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC
        lambda x: np.asfortranarray(x[:, mic_used]),
        lambda x: librosa.feature.melspectrogram(y=x, sr=sample_rate, n_mels=mel_banks, fmax=sample_rate // 2),
        lambda x: Tensor(x)
    ])

    # initialize the dataset
    my_dataset = PriusData(data_dir, transform=my_transforms, mode="static")

    # split the dataset
    train_data, val_data, test_data = stratified_split(my_dataset, mode="three_split", batch_size=batch_size)

    # setup the model, optimiser and the loss function
    my_model = CRNN(cnn_dropout=0.25).to(DEVICE)
    my_optim = torch.optim.Adam(my_model.parameters(), lr=0.001)
    my_loss  = BCEWithLogitsLoss()

    # training the model
    for epoch in tqdm(range(epochs), desc="Training the model"):

        # training the model
        my_model = my_model.train()

        my_model = run_one_epoch(model=my_model,
                                 data=train_data,
                                 loss_function=my_loss,
                                 optimizer=my_optim,
                                 device=DEVICE,
                                 epoch_num=epoch,
                                 mode="train",
                                 batch_size=batch_size)

        # check model against validation data
        my_model = my_model.eval()
        with no_grad():
            my_model = run_one_epoch(model=my_model,
                                     data=val_data,
                                     loss_function=my_loss,
                                     optimizer=None,
                                     device=DEVICE,
                                     epoch_num=epoch,
                                     mode="validation",
                                     batch_size=batch_size)


# set up dataloader for the entire dataset
# static_dataset = DataLoader(
#     my_dataset,
#     batch_size=1,
#     shuffle=True,
#     num_workers=1)
