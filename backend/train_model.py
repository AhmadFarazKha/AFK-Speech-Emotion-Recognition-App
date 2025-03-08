import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from utils.feature_extraction import extract_features

def train_emotion_model(data_path, save_path='models/emotion_model.h5'):
    """
    Train a speech emotion recognition model
    
    Args:
        data_path: Path to directory containing audio files with emotions as subdirectories
        save_path: Path to save trained model
        
    Returns:
        None
    """
    # Lists to store features and labels
    features = []
    labels = []
    
    # Dictionary of emotion categories
    emotions = {
        'angry': 0,
        'fear': 1,
        'happy': 2,
        'neutral': 3,
        'sad': 4,
        'surprise': 5
    }
    
    # Extract features from audio files
    for emotion, label in emotions.items():
        emotion_dir = os.path.join(data_path, emotion)
        if os.path.exists(emotion_dir):
            for file in os.listdir(emotion_dir):
                file_path = os.path.join(emotion_dir, file)
                if file.endswith('.wav'):
                    feature = extract_features(file_path)
                    if feature is not None:
                        features.append(feature)
                        labels.append(label)
    
    # Convert lists to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Convert labels to one-hot encoding
    labels = to_categorical(labels, num_classes=len(emotions))
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Build LSTM model
    model = Sequential([
        LSTM(128, input_shape=(features.shape[1], features.shape[2]), return_sequences=True),
        Dropout(0.4),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(len(emotions), activation='softmax')
    ])
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model on test data
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # You would need to provide the path to your emotion dataset
    # Example: train_emotion_model("path/to/emotion_dataset")
    print("Please specify the path to your emotion dataset to train the model.")