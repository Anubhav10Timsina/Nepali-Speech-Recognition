import os
import numpy as np
import joblib  # Added for saving scaler and label encoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler # Added StandardScaler
from config import PROCESSED_DATA_PATH, MODEL_SAVE_PATH, NUM_CLASSES

def load_data():
    X = np.load(os.path.join(PROCESSED_DATA_PATH, 'X.npy'))
    Y = np.load(os.path.join(PROCESSED_DATA_PATH, 'Y.npy'))
    
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)
    
    # SAVE LABEL ENCODER: So your prediction script knows what index means what word
    joblib.dump(le, 'label_encoder.joblib')
    print("Label Encoder saved as label_encoder.joblib")
    
    Y_categorical = to_categorical(Y_encoded, num_classes=NUM_CLASSES)
    return X, Y_categorical

def build_scratch_model(input_shape):
    model = Sequential()
    # Layer 1
    model.add(Dense(256, input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    # Layer 2
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    # Layer 3
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    # Output
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

def main():
    # 1. Load data
    print("Loading data...")
    X, Y = load_data()
    
    # 2. Process and Scale
    # Flatten X if it's 3D (samples, features, 1) to 2D (samples, features)
    if len(X.shape) == 3:
        X = X.reshape(X.shape[0], -1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SAVE SCALER: This fixes your "scaler is not defined" error in the other script
    joblib.dump(scaler, 'scaler.joblib')
    print("Scaler saved as scaler.joblib")

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
    
    # 3. Build & Compile
    print("Building custom model...")
    model = build_scratch_model(X_train.shape[1])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 4. Callbacks
    early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=0.00001, verbose=1)
    
    # 5. Train
    print("Starting training on Nepali dataset...")
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, lr_reduction]
    )
    
    # 6. Save Model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()