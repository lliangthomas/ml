import torch
import torch.nn as nn

class HybridModel(nn.Module):
    """
    A model comprised of LSTM and linear regression. Using basic EDA, I found that there was a somewhat linear correlation
    between time and number of receipts. Through some computational testing, I discovered purely using an LSTM would cause a 
    significant drop in the receipts, which was incorrect. Based on what we know, there should still be a general correlation
    between time/num receipts but it should still use past knowledge from the dataset beyond linear regression. Hence, a hybrid model. 
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_from_lstm: int = 2):
        super(HybridModel, self).__init__()
        
        # self.rnn = nn.LSTM(
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dtype=torch.float64
        )
        
        self.lstm_linear = nn.Linear(hidden_dim, num_from_lstm, dtype=torch.float64)
        # self.output_linear = nn.Linear(num_from_lstm + 1, 1, dtype=torch.float64)
        self.output_linear = nn.Linear(num_from_lstm + 1, 1, dtype=torch.float64)

    def forward(self, x):
        rnn_output, _ = self.rnn(x[:, :, :-1])
        linear_output = self.lstm_linear(rnn_output)
        days_since_start = x[:, :, -1].unsqueeze(-1)
        combine = torch.cat((linear_output, days_since_start), dim=-1)
        x = self.output_linear(combine)
        
        return x[:, -1]