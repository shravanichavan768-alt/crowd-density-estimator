import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import CSRNet
from dataset import ShanghaiTechDataset

DATA_PATH   = './data/ShanghaiTech'
PART        = 'B'          
BATCH_SIZE  = 8
EPOCHS      = 100
LR          = 1e-6
SAVE_PATH   = './models/csrnet_best.pth'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    print(f"Using device: {DEVICE}")

    
    train_dataset = ShanghaiTechDataset(DATA_PATH, part=PART, split='train')
    test_dataset  = ShanghaiTechDataset(DATA_PATH, part=PART, split='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=1,          shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # Model 
    model = CSRNet(load_weights=True).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_mae = float('inf')

    # Training Loop 
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0

        for images, density_maps in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            images      = images.to(DEVICE)
            density_maps = density_maps.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, density_maps)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Validation 
        mae, mse = evaluate(model, test_loader)

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | MAE: {mae:.2f} | MSE: {mse:.2f}")

        
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  Saved best model — MAE: {best_mae:.2f}")


def evaluate(model, loader):
    model.eval()
    mae_list = []
    mse_list = []

    with torch.no_grad():
        for images, density_maps in loader:
            images      = images.to(DEVICE)
            density_maps = density_maps.to(DEVICE)

            output = model(images)

            predicted_count = output.sum().item()
            actual_count    = density_maps.sum().item()

            diff = abs(predicted_count - actual_count)
            mae_list.append(diff)
            mse_list.append(diff ** 2)

    mae = np.mean(mae_list)
    mse = np.sqrt(np.mean(mse_list))

    return mae, mse


if __name__ == '__main__':
    os.makedirs('./models', exist_ok=True)
    train()