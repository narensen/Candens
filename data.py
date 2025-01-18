import pandas as pd
from torch.utils.data import Dataset
import os

df=pd.read_csv('/home/naren/Downloads/FacialBeauty/labels.txt',sep=' ',header=None)
df.columns=["filename", "rate"]

class Dataset(Dataset):

    def __init__(self, img : str, labels : str):
        super().__init__()

        self.img = Path(img)
        self.labels = Path(labels)

    def __concat__(self):


    def __getitem__(self, index):
        return super().__getitem__(index)



