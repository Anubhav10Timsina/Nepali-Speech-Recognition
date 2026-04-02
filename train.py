import os
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import PROCESSED_DATA_PATH, MODEL_SAVE_PATH, NUM_CLASSES

def load_data():
    X = np.load(os.path.join(PROCESSED_DATA_PATH, 'X.npy'))
    Y = np.load(os.path.join(PROCESSED_DATA_PATH, 'Y.npy'))
    
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)
    
    joblib.dump(le, 'label_encoder.joblib')
    print("Label Encoder saved as label_encoder.joblib")
    
    Y_categorical = to_categorical(Y_encoded, num_classes=NUM_CLASSES)
    return X, Y_categorical

def build_lstm_model(input_shape):
    """
    A robust Bidirectional LSTM architecture for sequence classification.
    input_shape should be (time_steps, features)
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # Layer 1: Bidirectional LSTM
        # return_sequences=True is required because the next layer is also an LSTM
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Layer 2: Second LSTM Layer
        # return_sequences=False because the next layer is Dense (not LSTM)
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        # Layer 3: Dense bottleneck
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Output Layer
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def main():
    # 1. Load data
    print("Loading data...")
    X, Y = load_data()
    
    # 2. Reshaping for LSTM
    # LSTMs expect (samples, time_steps, features). 
    # If your X is (samples, 130) or similar flattened MFCCs:
    if len(X.shape) == 2:
        # Example: 13 MFCCs over 10 time steps = 130 features.
        # Adjust the '10' and '13' below to match your actual feature extraction logic.
        # For now, I'll treat it as (samples, features, 1) to make it 3D.
        X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # 3. Scaling
    # Standard Scaler works on 2D, so we scale then reshape back
    samples, steps, features = X.shape
    X_reshaped = X.reshape(-1, features)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_final = X_scaled.reshape(samples, steps, features)

    joblib.dump(scaler, 'scaler.joblib')
    print("Scaler saved as scaler.joblib")

    X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y, test_size=0.2, random_state=42)
    
    # 4. Build and Compile
    print(f"Building LSTM model with input shape: {X_train.shape[1:]}")
    model = build_lstm_model(X_train.shape[1:])
    
    # Using a slightly smaller learning rate for LSTMs to maintain stability
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 5. Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5, min_lr=0.00001, verbose=1)
    
    # 6. Train
    print("Starting training...")
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, lr_reduction]
    )
    
    # 7. Save Model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()