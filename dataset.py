from torch.utils.data import Dataset
import numpy as np
import torch


class TestDataset(Dataset):
    def __init__(self, data, num_ent):
        super(TestDataset, self).__init__()
        self.data = data
        self.num_ent = num_ent

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item]
        triple, label = torch.tensor(item['triple'], dtype=torch.long), np.array(item['label'], dtype=np.int32)
        label = self.get_label(label)
        return triple, label

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.from_numpy(y)
