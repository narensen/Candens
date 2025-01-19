import time
import os

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


def dataset(device):
    print(device)
    df=pd.read_csv('/home/naren/Downloads/FacialBeauty/labels.txt',sep=' ',header=None)
    dir_images = "/home/naren/Downloads/FacialBeauty/Images/Images"
    df.columns=["filename", "rate"]
    df['image_path']=df['filename'].apply(lambda x:os.path.join(dir_images,x))

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


class BeautyDataset(Dataset):
    def __init__(self, df, transforms = None):

        self.df = df
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_scv = self.df.loc[idx, "image_path"]
        image = cv2.imread(image_scv, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = self.df.loc[idx, "rate"]
        labels = torch.from_numpy(labels.astype(np.float32))
        labels = labels.squeeze()

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
            
        
        return {'images':image,'labels':labels}






