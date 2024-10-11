import torch
import torch.nn as nn
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lstm_model import LSTMModel
from hybrid_model import HybridModel

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def window_dataset(data, window_size):
    x = []
    y = []
    for i in range(window_size, len(data)):
        x.append(data[i - window_size:i, 1:])
        y.append(data[i, 0])
    return torch.stack(x), torch.stack(y).reshape(-1, 1)

def preprocess_data(input_file, scale_range=None):
    df = pd.read_csv(input_file)
    df["# Date"] = pd.to_datetime(df["# Date"], yearfirst=True)
    # year as it is not a useful feature here because no data of year, 2022 would only destablize the model
    df['month'] = df['# Date'].dt.month
    df['day'] = df['# Date'].dt.day
    # df['quarter'] = df['date'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['days_since_start'] = (df['# Date'] - df['# Date'].dt.to_period('Y').min().to_timestamp()).dt.days
    data = torch.from_numpy(df.iloc[:, 1:].values.astype(np.float64)) # get rid of "# Date"
    if 'Receipt_Count' in df:
        nmin = torch.min(data[:, 0])
        nmax = torch.max(data[:, 0])
        # Numerical scaling for model stability
        data[:, 0] = (data[:, 0] - nmin) * (scale_range - 0) / (nmax - nmin)
        return data, nmin, nmax
    return data, df

def get_dataloader(scale_range: int, batch_size: int = 32, split: float = 1.0, window_size: int = 120):
    data, nmin, nmax = preprocess_data("data_daily.csv", scale_range)

    train_split = int(len(data) * split)
    train_data = data[:train_split - window_size]

    x_train, y_train = window_dataset(train_data, window_size)
    num_features = x_train.shape[-1] - 1 # subtract by 1 because days_since_start is only for linear regression

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), shuffle=True, batch_size=batch_size)

    return train_loader, nmin, nmax, num_features

def train(name: str, num_epochs: int = 1000, lr: float = 0.001, val_step: int = 100, scale_range: int = 100):
    train_loader, nmin, nmax, num_features = get_dataloader(scale_range)
    model = HybridModel(input_dim=num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            model.train()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f"Training Loss: {train_loss}")
        train_losses.append(train_loss)

    # torch.compile would not help here because no GPU :(
    torch.save({
        'model_state_dict': model.state_dict(),
        'nmin': nmin,
        'nmax': nmax,
        'scale_range': scale_range, 
    }, f'weights/{name}.pth')
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"plot_learning_curve/{name}.pdf")
    plt.close()

def inference(name: str, window_size: int = 120, plot=False):
    output = []
    gt_df = pd.read_csv("data_daily.csv")
    gt_df["# Date"] = pd.to_datetime(gt_df["# Date"], yearfirst=True)
    gt_df['month'] = gt_df['# Date'].dt.month
    gt_monthly = gt_df.groupby('month')["Receipt_Count"].sum().reset_index()

    infer_data, infer_df = preprocess_data("new_dates.csv")
    x = []
    for i in range(window_size, len(infer_data)):
        x.append(infer_data[i - window_size:i])
    infer_data = torch.stack(x)

    checkpoint = torch.load(f"weights/{name}.pth")

    model = HybridModel(infer_data.shape[-1] - 1) # subtract by 1 because days_since_start is only for linear regression
    model.load_state_dict(checkpoint['model_state_dict'])
    nmin, nmax, scale_range = checkpoint['nmin'], checkpoint['nmax'], checkpoint['scale_range']

    model.eval()
    with torch.no_grad():
        pred = model(infer_data)
        pred = pred * (nmax - nmin) / scale_range + nmin
    pred_df = pd.DataFrame({'date': pd.to_datetime(infer_df['# Date'][window_size:], yearfirst=True), 'prediction': pred.reshape(-1)})
    pred_df['month'] = pred_df['date'].dt.month
    pred_df['year'] = pred_df['date'].dt.year
    pred_df = pred_df[pred_df['year'] == 2022]
    # Aggregate for monthly predictions
    pred_monthly = pred_df.groupby('month')["prediction"].sum().reset_index()
    
    for i in range(1, 12 + 1):
        output.append({"date": f"{str(i).zfill(2)}/2021", "actual": gt_monthly.iloc[i - 1, 1].item()})
        output.append({"date": f"{str(i).zfill(2)}/2022", "prediction": pred_monthly.iloc[i - 1, 1].item()})
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(gt_df['# Date'], gt_df['Receipt_Count'], 'r-', label='Ground Truth')
        plt.plot(infer_df['# Date'][window_size:], pred, 'b-', label='Prediction')
        plt.xlabel('Date')
        plt.ylabel('Receipt Count')
        plt.savefig(f"plot_inference/{name}.pdf")
        plt.close()
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()
    if args.train:
        train(args.name)
    inference(args.name, plot=True)
    