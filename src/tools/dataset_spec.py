import os
import sys
import torch
import numpy as np
import pandas as pd

from torch import Tensor
from scipy.io import wavfile
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split

from .settings import SAMPLE_LENGTH, CLASS_MODE


class PriusData(Dataset):

    def __init__(self, root_dir,
                 transform=None,
                 mode="static",
                 class_type="coarse_class",
                 label_type="strong_label",
                 drop_front=False):
        self.data_dir = root_dir
        self.mode = mode
        self.class_type = class_type
        self.label_type = label_type
        self.transform = transform
        self.drop_front = drop_front

        # make it easier to read the csv file
        self.header_names = ["bag_name", "fine_class", "filename", "coarse_class", "location", "frame_num",
                             "ego_motion"]
        self.avail_class_t = ["coarse_class", "fine_class"]
        self.use_columns = [0, 1, 2, 3, 5, 6, 8]
        self.avail_mixed_modes = ["static", "driving", "all"]
        self.avail_label_t = ["strong_label", "weak_label"]
        self.avail_category_modes = ["SA", "SB", "SA1", "SA2", "SB1", "SB2", "SB3",
                                     "SB12", "SB23", "SB31", "DA", "DB"]
        self.location_labels = {"SA1": ["AnnaBoogerd"], "SA2": ["Kwekerij"], "SB1": ["Willem", "TanthofBackup"],
                                "SB2": ["Vermeerstraat"], "SB3": ["Geerboogerd"],
                                "SB12": ["Willem", "TanthofBackup", "Vermeerstraat"],
                                "SB23": ["Vermeerstraat", "Geerboogerd"],
                                "SB31": ["Geerboogerd", "Willem", "TanthofBackup"],
                                "DA": ["KwekerijD"],
                                "DB": ["VermeerstraatD"]}

        if self.mode not in (self.avail_mixed_modes + self.avail_category_modes):
            raise ValueError("Selected mode ({}) not available. Available modes: {}"
                             .format(self.mode, self.avail_mixed_modes + self.avail_category_modes))

        if len(self.header_names) != len(self.use_columns):
            raise ValueError("Number of columns({}) do not match the headers wanted({})".format(len(self.header_names),
                                                                                                len(self.use_columns)))

        self.csv_file = pd.read_csv(os.path.join(root_dir, "label_log_crnn.csv"), usecols=self.use_columns, header=None,
                                    names=self.header_names)

        if drop_front:
            self.csv_file = self.csv_file.drop(self.csv_file[self.csv_file["fine_class"] == "front"].index)

        # slice data as per the mode selected
        if self.mode == "static" or self.mode == "driving":
            is_mode = self.csv_file["ego_motion"] == self.mode
            self.rel_data = self.csv_file[is_mode]
        elif self.mode == "all":
            self.rel_data = self.csv_file
        elif self.mode in self.location_labels:
            self.rel_data = self.csv_file[self.csv_file["location"].isin(self.location_labels[self.mode])]
        elif self.mode in "SA":
            self.rel_data = self.csv_file[self.csv_file["location"].isin(self.location_labels["SA1"] +
                                                                         self.location_labels["SA2"])]
        elif self.mode in "SB":
            self.rel_data = self.csv_file[self.csv_file["location"].isin(self.location_labels["SB1"] +
                                                                         self.location_labels["SB2"] +
                                                                         self.location_labels["SB3"])]

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

        filename = self.rel_data["filename"].iloc[idx]
        sample_type = self.rel_data["fine_class"].iloc[idx]

        file_location = os.path.join(*[self.data_dir, sample_type, filename])

        s_rate, data = wavfile.read(file_location)

        if data.shape[0] == SAMPLE_LENGTH * s_rate:  # temporary fix - some samples are corrupted in engineOn-1-2-0.5
            if self.transform is not None:
                data = self.transform(data)
            else:
                data = data

            label = self.rel_data[self.class_type].iloc[idx]
            weak_label = label

            if self.label_type == "strong_label":
                label = self.get_strong_labels(weak_label, data.shape[1], mode=CLASS_MODE)
            elif self.label_type not in self.avail_label_t:
                raise ValueError("Selected label type ({}) not available. Choose from: {}"
                                 .format(self.label_type, self.avail_label_t))

            return data, label, weak_label
        else:
            return None

    def get_strong_labels(self, label, t_size, mode="multi_class"):
        """
        convert weak labels to strong.

        :param mode:
        :param t_size: temporal length of each spectrogram
        :param label: weak labels in the form of str
        :return: strong labels
        """
        label_tensor = torch.ones([1, t_size], dtype=torch.float64)
        strong_label = None

        if mode == "multi_class":
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
        elif mode == "single_class":
            if self.class_type == "fine_class":
                raise ValueError ("Cannot have mode: {} and class division: {} together".format(mode, self.class_type))
            elif self.class_type == "coarse_class":
                strong_label = torch.zeros([1, t_size], dtype=torch.float64)
                if label == "positive":
                    strong_label = label_tensor

            return strong_label

        else:
            raise ValueError("Mode: {}, unknown. Choose from : {}".format(mode, ["multi_class", "single_class"]))


class JoinDataset(PriusData):

    def __init__(self, dataset_1, dataset_2):
        super().__init__(dataset_1.data_dir)

        self.rel_data = pd.concat([dataset_1.rel_data, dataset_2.rel_data])
        self.targets = pd.concat([dataset_1.targets, dataset_2.targets])
        self.transform = dataset_1.transform


class BagSplitData(PriusData):

    def __init__(self, dataset, test_split=0.2, mode="train", seed=42, drop_front=False):
        super().__init__(dataset.data_dir, drop_front=drop_front)

        # True if random samples needs to be dropped from positive class for fair comparison when drop_front=False
        equalize_train = False

        unique_bags = dataset.rel_data.drop_duplicates(subset=["bag_name"])[["bag_name", "location"]]

        train_bags, test_bags = train_test_split(list(unique_bags["bag_name"]), test_size=test_split, shuffle=True,
                                                 stratify=list(unique_bags["location"]), random_state=seed)

        if mode == "train":
            self.rel_data = dataset.rel_data[dataset.rel_data["bag_name"].isin(train_bags)]
            if drop_front:
                self.rel_data = self.rel_data.drop(self.rel_data[self.rel_data["fine_class"] == "front"].index)
            else:
                if equalize_train:
                    num_front = len(self.rel_data[self.rel_data["fine_class"] == "front"])
                    temp_df = self.rel_data[self.rel_data["fine_class"].isin(["front", "left", "right"])]
                    temp_df_idxs = temp_df.index
                    to_remove = np.random.choice(temp_df_idxs, size=num_front, replace=False)
                    temp_df = temp_df.drop(to_remove)
                    self.rel_data = self.rel_data.drop(temp_df_idxs)
                    self.rel_data = pd.concat([self.rel_data, temp_df])

        elif mode == "test":
            self.rel_data = dataset.rel_data[dataset.rel_data["bag_name"].isin(test_bags)]
            if drop_front:
                self.rel_data = self.rel_data.drop(self.rel_data[self.rel_data["fine_class"] == "front"].index)
        else:
            raise ValueError("Selected mode ({}) not available. Choose from: {}".format(mode, ["train", "test"]))

        self.targets = self.rel_data[dataset.class_type]
        self.transform = dataset.transform

        print(list(set(self.rel_data["fine_class"])))