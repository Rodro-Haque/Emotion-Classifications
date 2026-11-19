# Multimodal Emotion Classification

A multimodal emotion classification project combining text, audio, and visual features using neural encoders and fusion models in PyTorch.

## Repository Contents

- **`mid_final/`** — Multi-label multimodal emotion classification pipeline using the CMU-MOSEI dataset (6 emotion classes: `happy`, `sad`, `anger`, `surprise`, `disgust`, `fear`).
- **`MultimodalAI_mid/`** — Feature-extraction and early-fusion multimodal pipeline (7 classes; DistilBERT text, MFCC audio, and ResNet18 visual features).
- **`Multimodal Emotion Classification.pptx`** — Project presentation slides.

---

## `mid_final/` Setup & Usage (CMU-MOSEI Multi-Label)

1. **Create a Python 3.10 virtual environment and install dependencies:**
   ```bash
   cd mid_final
   python -m venv mult_ai
   mult_ai\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add CMU-MOSEI `.csd` dataset files** into `mid_final/data/csd/`:
   - `CMU_MOSEI_Labels.csd`
   - `CMU_MOSEI_TimestampedWordVectors.csd`
   - `CMU_MOSEI_COVAREP.csd`
   - `CMU_MOSEI_VisualOpenFace2.csd`

3. **Prepare the dataset:**
   ```bash
   python prepare_mosei_multilabel_from_csd.py --csd_dir data/csd --out_dir data/processed
   ```

4. **Update `mid_final/config.py`** using the dimensions generated in `data/processed/feature_dims.json` if needed, then train and evaluate:
   ```bash
   python train.py
   python evaluate.py
   ```

---

## `MultimodalAI_mid/` Setup & Usage

1. Download the dataset referenced in `MultimodalAI_mid/data/source.txt` into `MultimodalAI_mid/data/` (keeping the `data/processed/` directory intact).
2. Run the pipeline steps described in `MultimodalAI_mid/instructions.txt`:
   ```bash
   cd MultimodalAI_mid
   python build_metadata.py
   python split_data.py
   python scripts/extract_text_features.py
   python scripts/extract_audio_features.py
   python scripts/extract_visual_features.py
   python train.py
   python evaluate.py
   ```
