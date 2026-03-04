import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
from constants import MODEL_PATH, FEAT_PATH, TRAIN_PATH, VALID_PATH
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Loaded device: {device}")

class NumeraiNet(nn.Module):
    def __init__(self, input_size):
        super(NumeraiNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.layers(x))

def train():
    with open(FEAT_PATH, 'r') as f:
        features = json.load(f)['feature_sets']['medium']

    df = pd.read_parquet(TRAIN_PATH, columns=['target'] + features).dropna()
    df_val = pd.read_parquet(VALID_PATH, columns=['target'] + features).dropna()

    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(df['target'].values, dtype=torch.float32).view(-1, 1)
    
    X_val = torch.tensor(df_val[features].values, dtype=torch.float32).to(device)
    y_val = df_val['target'].values 

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)

    model = NumeraiNet(len(features)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    criterion = nn.MSELoss()

    epochs = 10 
    best_corr = -1.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).cpu().detach().flatten().tolist()
        current_corr = pd.Series(y_val).corr(pd.Series(val_preds), method='spearman')

    print(f"Epoch {epoch+1} completed, Avg Loss: {epoch_loss/len(loader):.6f}, Validate CORR: {current_corr:.4f}")

    if current_corr > best_corr:
        best_corr = current_corr
        torch.save(model.state_dict(), MODEL_PATH[1])
        print(f"New optimized model saved at {MODEL_PATH[1]} (Best CORR: {best_corr:.4f})")

if __name__ == "__main__":
    train()