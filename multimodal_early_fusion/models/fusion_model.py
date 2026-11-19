import torch
import torch.nn as nn
import config
from models.text_encoder import TextEncoder
from models.audio_encoder import AudioEncoder
from models.visual_encoder import VisualEncoder

class EarlyFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.audio_encoder = AudioEncoder()
        self.visual_encoder = VisualEncoder()

        fusion_dim = config.TEXT_HIDDEN + config.AUDIO_HIDDEN + config.VISUAL_HIDDEN

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, config.FUSION_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FUSION_HIDDEN, config.NUM_CLASSES)
        )

    def forward(self, text, audio, visual):
        t = self.text_encoder(text)
        a = self.audio_encoder(audio)
        v = self.visual_encoder(visual)

        fused = torch.cat([t, a, v], dim=-1)
        logits = self.classifier(fused)
        return logits