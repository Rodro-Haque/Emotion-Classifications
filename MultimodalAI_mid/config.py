NUM_CLASSES = 7

TEXT_DIM = 768          # DistilBERT hidden size
AUDIO_DIM = 160         # 40 MFCC mean + 40 MFCC std + 40 delta mean + 40 delta std
VISUAL_DIM = 512        # ResNet18 pooled feature

TEXT_HIDDEN = 256
AUDIO_HIDDEN = 128
VISUAL_HIDDEN = 128
FUSION_HIDDEN = 256

DROPOUT = 0.3
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 20