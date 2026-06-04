"""Dataset wrapper used by the projection experiments.

Migrated verbatim (behaviour-preserving) from utils.general_utils.MyDatasetV2 +
collate_data. batch_size=1 iteration yields (prompt, label_idx) in row order.
"""
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def collate_data(xs):
    clean, correct_idx = zip(*xs)
    clean = list(clean)
    return clean, correct_idx


class MyDatasetV2(Dataset):
    def __init__(self, data, num_samples=None):
        self.df = pd.DataFrame(data)
        if num_samples is not None and num_samples > 0:
            self.df = self.df.sample(n=num_samples, random_state=20)

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row["prompt"], row["label_idx"]

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_data)
