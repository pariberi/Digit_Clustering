from torch.utils import data


class TransformDataset(data.Dataset):
    def __init__(self, data_tuple, device):
        self.device = device
        self.x, self.y = data_tuple
        self.x.to(self.device)
        self.x, self.y = self.x.to(self.device), self.y.to(self.device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], idx
