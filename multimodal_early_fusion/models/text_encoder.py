import torch.nn as nn
import config

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.TEXT_DIM, config.TEXT_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

    def forward(self, x):
        return self.net(x)