from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from typing import List
from pydantic import BaseModel

from torch.utils.data import DataLoader
import torch

import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data import BeautyDataset
from data import dataset
from train import fit

SIZE = 224
device = "cuda" if torch.cuda.is_available() else "cpu"

train_df, test_df = dataset(device)
transforms = A.Compose([
    A.Resize(height=SIZE, width=SIZE, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0)
])  

train_df = BeautyDataset(train_df, transforms)
test_df = BeautyDataset(test_df, transforms)

train_dataset = DataLoader(train_df, batch_size=32, shuffle=True, num_workers=4)
test_dataset = DataLoader(test_df, batch_size=32, shuffle=False, num_workers=4)

model = fit(train_dataset, test_dataset, device)





    

        


