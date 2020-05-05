import os
import sys
import torch
import numpy as np
import pandas as pd

from torch import Tensor
from scipy.io import wavfile
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class PriusData(Dataset):

    def __init__(self, root_dir, transform=None, mode="static", class_type="coarse_class"):

        self.data_dir = root_dir
        self.mode     = mode

        # make it easier to read the csv file
        header_names  = ["bag_name", "fine_class", "filename", "coarse_class", "location", "frame_num", "ego_motion"]
        avail_class_t = ["coarse_class", "fine_class"]
        use_columns   = [0,1,2,3,5,6,8]
        avail_modes   = ["static", "driving", "all"]

        if len(header_names)!=len(use_columns):
            raise ValueError("Number of columns({}) do not match the headers wanted({})".format(len(header_names), len(use_columns)))

        self.csv_file = pd.read_csv(os.path.join(root_dir, "label_log.csv"), usecols=use_columns, header=None, names=header_names)

        # slice data as per the mode selected
        if self.mode == "static" or self.mode == "driving":
            is_mode       = self.csv_file["ego_motion"] == self.mode
            self.rel_data = self.csv_file[is_mode]

        elif self.mode == "all":
            self.rel_data = self.csv_file

        else:
            raise ValueError("Selected mode ({}) not available. Available modes: {}".format(self.mode, avail_modes))

        self.transform = transform
        self.locations = list(set(self.rel_data["location"]))

        if class_type in avail_class_t:
            self.targets = self.rel_data[class_type]
            self.classes = list(set(self.targets))

        else:
            raise ValueError("Selected class type ({}) not available. Choose from: {}".format(class_type, avail_class_t))

    def __len__(self):

        return self.rel_data.shape[0]

    def __getitem__(self, idx):

        filename    = self.rel_data["filename"][idx]
        sample_type = self.rel_data["fine_class"][idx]

        file_location = os.path.join(*[self.data_dir, sample_type, filename])

        s_rate, data = wavfile.read(file_location)

        if self.transform is not None:
            data = self.transform(data)
        else:
            data = data

        label = self.rel_data["coarse_class"][idx]

        return data, label







