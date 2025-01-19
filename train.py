import time
from model import Candens
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from tqdm import tqdm


def fit(dataloader, test_dataloader, device="cpu", epochs=15):

    model = Candens().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()    

    for epoch in range(epochs+1):
        train_loss = 0
        val_loss = 0
        train_time = time.time
        print(f"{epoch+1}/{epochs} Epochs")

        model.train()
        with tqdm(dataloader, unit="batch") as tepoch:
            for _, d in enumerate(dataloader):

                inputs = d["images"].to(device)
                labels = d["labels"].to(device)
                
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                tepoch.set_postfix(loss=train_loss / (_ + 1))

        epoch_time = time.time() - train_time

        print(f"Training Loss: {train_loss/len(dataloader):.4f} Time:{epoch_time:.2f}s")

        model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            for _, d in enumerate(test_dataloader):

                inputs = d["images"].to(device)
                labels = d["labels"].to(device)

                output = model(inputs)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(output.cpu().numpy())

            r2 = r2_score(y_true, y_pred)
            print(r2)

            if r2 > best_accuracy:
                best_accuracy = r2
                if r2 >= 85:
                    torch.save(model.state_dict(), "best_model.pth")

    
    return model




            


            

