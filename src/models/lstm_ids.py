import torch
import torch.nn as nn

class LSTMIDS(nn.Module):

    def __init__(
        self,
        input_size=42,
        hidden_size=64,
        num_layers=1,
        dropout=0.3,
        num_classes=10,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(
            hidden_size,
            32,
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            32,
            num_classes,
        )

    def forward(self, x):

        # x:
        # [batch, sequence_length, features]

        lstm_out, _ = self.lstm(x)

        # Use the final timestep.
        last_output = lstm_out[:, -1, :]

        x = self.dropout(
            last_output
        )

        x = self.fc1(x)

        x = self.relu(x)

        logits = self.fc2(x)

        return logits