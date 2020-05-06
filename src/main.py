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
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

writer = SummaryWriter()

if __name__ == '__main__':

    data_dir = "../../data"
    sample_rate = 47998
    mel_banks = 128
    mic_used = 5  # mic number used
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    batch_size = 4

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
    my_model = CRNN().to(DEVICE)
    my_optim = torch.optim.Adam(my_model.parameters(), lr=0.001)
    my_loss  = BCEWithLogitsLoss()

    for epoch in tqdm(range(epochs)):

        # training the model
        my_model = my_model.train()

        training_loss = 0.0
        for idx, (sample, label) in enumerate(train_data):
            # debug only
            # print("{}: DataShape: {}, LabelShape: {}, {}".format(idx, sample.shape, label))

            # setting all gradients to zero, apparently useful,
            # see: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            my_optim.zero_grad()

            if DEVICE == torch.device("cuda"):
                sample = sample.cuda()
                label  = label.cuda()

            label_hat = my_model(sample)

            # compute loss and backpropagate
            loss = my_loss(label_hat, label)
            loss.backward()
            my_optim.step()

            # TODO: not sure if I have to clip the gradients here.

            training_loss += loss.item() * batch_size

            writer.add_scalar('Training Loss', training_loss, epoch*len(train_data)+idx)

            training_loss = 0.0





# set up dataloader for the entire dataset
# static_dataset = DataLoader(
#     my_dataset,
#     batch_size=1,
#     shuffle=True,
#     num_workers=1)
