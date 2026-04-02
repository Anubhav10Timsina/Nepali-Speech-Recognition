import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROCESSED_DATA_PATH = '/Users/ayamkattel/Desktop/MISC_LATEST/Nepali-Speech-Recognition/data/processed' 
MODEL_SAVE_PATH = 'nepali_lstm_model.h5'
NUM_CLASSES = 8

def load_data():
    """Consistent with Kaggle label encoding and saving"""
    X = np.load(os.path.join(PROCESSED_DATA_PATH, 'X.npy')) # Expected shape: (Samples, 182, 15)
    Y = np.load(os.path.join(PROCESSED_DATA_PATH, 'Y.npy'))

    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)
    
    # Save the encoder for inference later
    joblib.dump(le, 'label_encoder.joblib')
    print("Label Encoder saved as label_encoder.joblib")
    
    Y_categorical = to_categorical(Y_encoded, num_classes=NUM_CLASSES)
    return X, Y_categorical

def build_lstm_model(input_shape):
   
    model = Sequential([
        Input(shape=input_shape),
        
    
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2), 
        
        LSTM(64, return_sequences=True), 
        
        GlobalAveragePooling1D(),
        
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def main():
    print("Loading data...")
    X, Y = load_data()

    samples, time_steps, features = X.shape
    X_reshaped = X.reshape(-1, features)
    
    scaler = StandardScaler()
    X_scaled_reshaped = scaler.fit_transform(X_reshaped)
    
    joblib.dump(scaler, 'scaler.joblib')
    print("Scaler saved as scaler.joblib")

    X_final = X_scaled_reshaped.reshape(samples, time_steps, features)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_final, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    print(f"Building LSTM model with input shape: {X_train.shape[1:]}")
    model = build_lstm_model(X_train.shape[1:])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=6, factor=0.5, min_lr=0.00001, verbose=1)
    
    print("Starting training...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, lr_reduction]
    )
    
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()