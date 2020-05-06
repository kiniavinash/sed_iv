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

    def __init__(self, root_dir, transform=None, mode="static", class_type="coarse_class", label_type="strong_label"):

        self.data_dir = root_dir
        self.mode = mode
        self.class_type = class_type
        self.label_type = label_type

        # make it easier to read the csv file
        self.header_names = ["bag_name", "fine_class", "filename", "coarse_class", "location", "frame_num",
                             "ego_motion"]
        self.avail_class_t = ["coarse_class", "fine_class"]
        self.use_columns = [0, 1, 2, 3, 5, 6, 8]
        self.avail_modes = ["static", "driving", "all"]
        self.avail_label_t = ["strong_label", "weak_label"]

        if len(self.header_names) != len(self.use_columns):
            raise ValueError("Number of columns({}) do not match the headers wanted({})".format(len(self.header_names),
                                                                                                len(self.use_columns)))

        self.csv_file = pd.read_csv(os.path.join(root_dir, "label_log.csv"), usecols=self.use_columns, header=None,
                                    names=self.header_names)

        # slice data as per the mode selected
        if self.mode == "static" or self.mode == "driving":
            is_mode = self.csv_file["ego_motion"] == self.mode
            self.rel_data = self.csv_file[is_mode]

        elif self.mode == "all":
            self.rel_data = self.csv_file

        else:
            raise ValueError(
                "Selected mode ({}) not available. Available modes: {}".format(self.mode, self.avail_modes))

        self.transform = transform
        self.locations = list(set(self.rel_data["location"]))

        if class_type in self.avail_class_t:
            self.targets = self.rel_data[class_type]
            self.classes = list(set(self.targets))

        else:
            raise ValueError("Selected class type ({}) not available. Choose from: {}"
                             .format(class_type, self.avail_class_t))

    def __len__(self):

        return self.rel_data.shape[0]

    def __getitem__(self, idx):

        filename = self.rel_data["filename"][idx]
        sample_type = self.rel_data["fine_class"][idx]

        file_location = os.path.join(*[self.data_dir, sample_type, filename])

        s_rate, data = wavfile.read(file_location)

        if self.transform is not None:
            data = self.transform(data)
        else:
            data = data

        label = self.rel_data["coarse_class"][idx]

        if self.label_type == "strong_label":
            label = self.get_strong_labels(label, data.shape[1])

        elif self.label_type not in self.avail_label_t:
            raise ValueError("Selected label type ({}) not available. Choose from: {}"
                             .format(self.label_type, self.avail_label_t))

        return data, label

    def get_strong_labels(self, label, t_size):
        """
        convert weak labels to strong.
        :param label: weak labels in the form of str
        :return: strong labels
        """
        label_tensor = torch.ones([1, t_size], dtype=torch.float64)

        if self.class_type == "coarse_class":
            strong_label = torch.zeros([2, t_size], dtype=torch.float64)

            if label == "positive":
                strong_label[0, :] = strong_label[0, :] + label_tensor

            else:
                strong_label[1, :] = strong_label[1, :] + label_tensor

        elif self.class_type == "fine_class":
            strong_label = torch.zeros([4, t_size], dtype=torch.float64)

            if label == "front":
                strong_label[0, :] = strong_label[0, :] + label_tensor

            elif label == "left":
                strong_label[1, :] = strong_label[1, :] + label_tensor

            elif label == "negative":
                strong_label[2, :] = strong_label[2, :] + label_tensor

            elif label == "right":
                strong_label[3, :] = strong_label[3, :] + label_tensor

        return strong_label
