import os
import librosa
import numpy as np
from config import DATA_PATH, PROCESSED_DATA_PATH, SENTENCES, SAMPLE_RATE, DURATION

OFFSET = 0.0 
FRAME_LENGTH = 2048
HOP_LENGTH = 512

import os
import librosa
import numpy as np


FRAME_LENGTH = 2048
HOP_LENGTH = 512

def process_audio_sequence(audio, sample_rate):
    audio, _ = librosa.effects.trim(audio, top_db=20)
    if len(audio) > 0:
        audio = librosa.util.normalize(audio)
        
    target_length = int(sample_rate * DURATION)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, max(0, target_length - len(audio))), "constant")
    else:
        audio = audio[:target_length]
    
    zcr_val = librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH).T
    rmse_val = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH).T
    mfcc_val = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13, hop_length=HOP_LENGTH).T
    
    sequence = np.hstack((zcr_val, rmse_val, mfcc_val))
    return sequence

def process_and_save():
    X = []
    Y = []
    
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    print("Extracting sequential features for LSTM...")
    
    for category in SENTENCES:        
        folder_path = os.path.join(DATA_PATH, category)
        if not os.path.isdir(folder_path):
            print(f"Skipping {category}: directory not found")
            continue
            
        print(f"Processing category: {category}")
        count = 0
        for file in os.listdir(folder_path):
            if file.lower().endswith('.wav'): 
                file_path = os.path.join(folder_path, file)
                try:
                    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    features = process_audio_sequence(audio, sr)
                    
                    X.append(features)
                    Y.append(category)
                    count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
        print(f" -> {count} samples generated.")

    X = np.array(X)
    Y = np.array(Y)
    
    if len(X) > 0:
        print(f"\nFinal feature shape (Should be N, 182, 15): {X.shape}")
        
        np.save(os.path.join(PROCESSED_DATA_PATH, 'X.npy'), X)
        np.save(os.path.join(PROCESSED_DATA_PATH, 'Y.npy'), Y)
        print("Done saving new X.npy and Y.npy for the LSTM model!")
    else:
        print("No data extracted. Please check your config paths and audio files.")

if __name__ == "__main__":
    process_and_save()