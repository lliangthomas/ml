import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 128, 
                 hidden_dim3: int = 256, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            batch_first=True,
            dtype=torch.float64
        )
        
        self.rnn2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            batch_first=True,
            dtype=torch.float64
        )
        
        self.linear1 = nn.Linear(hidden_dim2, hidden_dim3, dtype=torch.float64)
        self.linear2 = nn.Linear(hidden_dim3, 1, dtype=torch.float64)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x[:, -1]