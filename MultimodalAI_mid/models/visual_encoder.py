import torch.nn as nn
import config

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.VISUAL_DIM, config.VISUAL_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

    def forward(self, x):
        return self.net(x)