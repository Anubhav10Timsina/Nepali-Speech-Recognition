import os
import librosa
import numpy as np
from config import DATA_PATH, PROCESSED_DATA_PATH, SENTENCES, SAMPLE_RATE, DURATION

# Audio parameters
OFFSET = 0.0 # No offset since the sentence audio starts immediately
FRAME_LENGTH = 2048
HOP_LENGTH = 512

def zcr(data, frame_length=2048, hop_length=512):
    zcr_val = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr_val)

def rmse(data, frame_length=2048, hop_length=512):
    rmse_val = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse_val)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)

def process_audio_array(audio, sample_rate):
    """Processes raw audio arrays to extract and stack features."""
    # 1. Padding / Truncation
    target_length = int(sample_rate * DURATION)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, max(0, target_length - len(audio))), "constant")
    else:
        audio = audio[:target_length]
    
    # 2. Extract the 3 features
    res1 = zcr(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    res2 = rmse(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    res3 = mfcc(audio, sr=sample_rate, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    
    # 3. Stack horizontally
    result = np.hstack((res1, res2, res3))
    return result

def main():
    X = []
    Y = []
    
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    print(f"Extracting features, categories to process: {len(SENTENCES)}")
    
    for category in SENTENCES:        
        folder_path = os.path.join(DATA_PATH, category)
        if not os.path.isdir(folder_path):
            print(f"Skipping {category}: directory not found")
            continue
            
        print(f"\nProcessing category: {category}")
        count = 0
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                try:
                    # Load original
                    audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, offset=OFFSET)
                    
                    # Process and append only the original audio
                    X.append(process_audio_array(audio, sample_rate))
                    Y.append(category)
                    count += 1
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
        print(f" -> Generated {count} samples for {category}")

    X = np.array(X)
    Y = np.array(Y)
    
    print(f"\nTotal dataset size extracted: {len(X)}")
    if len(X) > 0:
        X = np.expand_dims(X, axis=2)
        print(f"Final feature shape (ready for model): {X.shape}")
        
        np.save(os.path.join(PROCESSED_DATA_PATH, 'X.npy'), X)
        np.save(os.path.join(PROCESSED_DATA_PATH, 'Y.npy'), Y)
        print("Done saving X.npy and Y.npy")
    else:
        print("No data extracted. Please check your config paths and audio files.")

if __name__ == "__main__":
    main()