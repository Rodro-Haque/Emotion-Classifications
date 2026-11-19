# CMU-MOSEI multi-label emotion classification config

# CMU-MOSEI emotion labels: multi-label binary targets
EMOTION_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
NUM_CLASSES = len(EMOTION_NAMES)

# IMPORTANT:
# Update these after running prepare_mosei_multilabel_from_csd.py.
# The script creates data/processed/feature_dims.json.
TEXT_DIM = 300
AUDIO_DIM = 74
VISUAL_DIM = 713

TEXT_HIDDEN = 256
AUDIO_HIDDEN = 128
VISUAL_HIDDEN = 128
FUSION_HIDDEN = 256

DROPOUT = 0.3
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 30

# Threshold used during evaluation to convert sigmoid probabilities to 0/1 labels
THRESHOLD = 0.5
