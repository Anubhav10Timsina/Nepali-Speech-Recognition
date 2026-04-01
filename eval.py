import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import PROCESSED_DATA_PATH, MODEL_SAVE_PATH, SENTENCES

def load_validation_data():
    """Load the validation data."""
    X = np.load(os.path.join(PROCESSED_DATA_PATH, 'X.npy'))
    Y = np.load(os.path.join(PROCESSED_DATA_PATH, 'Y.npy'))
    
    # Scale X exactly like in train.py
    nsamples, nx, ny = X.shape
    X_reshaped = X.reshape((nsamples, nx * ny))
    scaler = StandardScaler()
    X = scaler.fit_transform(X_reshaped)
    
    # Convert labels to integers
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    
    return X, Y, le

def evaluate_model():
    """Evaluate the trained model on the validation set."""
    # Load validation data
    X, Y, label_encoder = load_validation_data()
    
    # Load the trained model
    model = load_model(MODEL_SAVE_PATH)
    print(f"Loaded model from {MODEL_SAVE_PATH}")
    
    # Make predictions
    predictions = model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(Y, predicted_classes, target_names=SENTENCES))
    
    # Generate confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(Y, predicted_classes))

if __name__ == "__main__":
    evaluate_model()