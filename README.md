# Nepali Speech Recognition

A Nepali sentence classification system that recognizes 8 fixed Nepali phrases from audio recordings using a Bidirectional LSTM neural network. The project includes a complete pipeline for data collection, augmentation, feature extraction, model training, live inference, and visualization.

## Target Sentences

| Label | Nepali Sentence | English Translation |
|-------|----------------|---------------------|
| 1 | Ma Khusi Xu | I am happy |
| 2 | Malai Mero Desh Pyaro Lagxa | I love my country |
| 3 | Namaste | Hello |
| 4 | Ram Le Vaat Khanxa | Ram eats food |
| 5 | Tapaiko Ghar Kaha Xa | Where is your house? |
| 6 | TimiLai Kasto Chha | How are you? |
| 7 | Uh Mero Mitra Ho | He is my friend |
| 8 | Yo Hamro AI Ko Project Ho | This is our AI project |

## Pipeline Overview

```
Data Collection → Preprocessing → Training → Inference
(collect_data)    (data_prep.py)  (train.py)  (predict.ipynb)
```

1. **Collect** — Record audio via interactive notebook; each recording is augmented into 11 variants (noise, pitch shift, time stretch, volume change)
2. **Preprocess** — Trim silence, normalize, extract 15 features per frame (ZCR + RMS + 13 MFCCs) → shape `(182, 15)` per sample
3. **Train** — Bidirectional LSTM (~99K params) with EarlyStopping and learning rate scheduling
4. **Predict** — Live microphone recording → feature extraction → classification with confidence score

## Model Architecture

```
Input (182, 15) → Bidirectional LSTM (64) → BatchNorm → Dropout(0.2)
    → LSTM (64) → GlobalAveragePooling1D → BatchNorm
    → Dense (64, ReLU) → Dropout(0.2) → Dense (8, Softmax)
```

- ~99K parameters
- Achieves ~100% validation accuracy on the 8-class task
- Converges in ~33 epochs with early stopping

## Project Structure

```
├── config.py                   # Paths, hyperparameters, class labels
├── utils.py                    # Utility functions
├── data_prep.py                # Feature extraction pipeline
├── train.py                    # Model definition and training
├── collect_data.ipynb          # Interactive data collection + augmentation
├── predict.ipynb               # Live inference and test evaluation
├── visualize.ipynb             # Data visualization and analysis
├── speakers.csv                # Speaker list (20 speakers)
├── nepali_model_artifacts/     # Pre-trained model, scaler, encoder
│   ├── nepali_lstm_model.h5
│   ├── scaler.joblib
│   └── label_encoder.joblib
├── test/                       # Test dataset (raw audio + processed features)
├── report/                     # LaTeX report
└── misc/                       # Alternative Kaggle notebook and backups
```

## Getting Started

### Prerequisites

```bash
pip install tensorflow librosa numpy scikit-learn joblib scipy
pip install sounddevice soundfile ipywidgets   # for data collection
pip install pyaudio                             # for live inference
pip install matplotlib seaborn pandas           # for visualization
```

### 1. Collect Data

```bash
jupyter notebook collect_data.ipynb
```

Record audio for each of the 8 sentences. Each recording produces 11 files (1 original + 10 augmented).

### 2. Preprocess

```bash
python data_prep.py
```

Extracts features from `data/raw/` and saves `X.npy`, `Y.npy` to `data/processed/`.

### 3. Train

```bash
python train.py
```

Trains the LSTM model and saves `nepali_lstm_model.h5`, `scaler.joblib`, and `label_encoder.joblib`.

### 4. Predict

```bash
jupyter notebook predict.ipynb
```

Records live audio from the microphone and outputs the predicted sentence with confidence.

### 5. Visualize

```bash
jupyter notebook visualize.ipynb
```

Generates class distribution plots, audio feature visualizations (waveform, spectrogram, MFCC), and confusion matrix.

## Dataset

- **Format**: WAV, mono, 22050 Hz
- **Speakers**: 20
- **Samples**: ~1,221 (after 11x augmentation)
- **Structure**: `data/raw/{Sentence_Label}/spk{id}_s{id}.wav`
- **Augmentations**: Gaussian noise, pitch shift (+-2 semitones), time stretch (0.8x/1.2x), volume scaling, combined variations

## Feature Engineering

Each audio clip (4.224s) is converted to a `(182, 15)` feature matrix:

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| ZCR | 1 | Zero-Crossing Rate |
| RMS | 1 | Root Mean Square Energy |
| MFCC | 13 | Mel-Frequency Cepstral Coefficients |

Frame length: 2048 samples (~92ms), Hop length: 512 samples (~23ms)

## Technical Report

See [CODEBASE_REPORT.md](CODEBASE_REPORT.md) for detailed technical documentation of the entire codebase.
