import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np
import os

# Create a simple LSTM model similar to what we would train
model = Sequential([
    LSTM(128, input_shape=(174, 40), return_sequences=True),
    Dropout(0.4),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(6, activation='softmax')  # 6 emotions
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Save the model
os.makedirs('models', exist_ok=True)
model.save('models/emotion_model.h5')

print("Placeholder model created and saved to models/emotion_model.h5")