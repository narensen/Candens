import time
from model import Candens
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


def fit(dataloader, device, epochs=15):

    model = Candens().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()    

    for epoch in range(epochs+1):
        train_loss = 0
        val_loss = 0
        train_time = time.time
        print(f"{epoch+1}/{epochs} Epochs")

        model.train()
        for _, d in enumerate(dataloader):

            inputs = d["images"].to(device)
            labels = d["labels"].to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        epoch_time = time.time() - train_time
        print(f"Training Loss: {train_loss/len(dataloader):.4f} Time:{epoch_time:.2f}s")
    
    return model




            


            

