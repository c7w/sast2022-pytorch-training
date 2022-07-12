import torch
import numpy as np
from PIL import Image
import torch.utils.data.Dataset as Dataset


class LandScapeDataset(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if mode == "train" or mode == "val":
            with open(f"../data/{mode}/file.txt", 'r') as file:
                entries = file.read().strip().split('\n')[1:]
                table = [entry.split(",") for entry in entries]
                self.images = [row[0] for row in table]
                self.gt = [(bool(row[1]), bool(row[2]), bool(row[3])) for row in table]
        elif mode == "test":
            with open(f"../data/{mode}/file.txt", 'r') as file:
                self.images = file.read().strip().split('\n')[1:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        :param idx: index to get.
        :return: ret_dict: {"image": torch.tensor, "label": torch.tensor}
        """

        file_name = self.images[idx]

        ret_dict = {
            "image": torch.tensor(np.array(Image.open(f"../data/{self.mode}/imgs/{file_name}"))),
        }

        if self.mode != "test":
            ret_dict["label"] = torch.tensor(np.array(self.gt[idx]))

        return ret_dict
