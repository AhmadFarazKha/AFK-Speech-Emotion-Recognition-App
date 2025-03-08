from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import tempfile
from utils.feature_extraction import extract_features, extract_features_live
from utils.preprocessing import preprocess_audio, convert_audio_format

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the emotion recognition model
model = None
try:
    model = tf.keras.models.load_model('models/emotion_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    
# Emotion labels
EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# For testing without a model
USE_DUMMY_PREDICTION = False

@app.route('/')
def index():
    return jsonify({
        "status": "active",
        "message": "AFK Niazi Speech Emotion Recognition API is running"
    })

@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        # Check if a file was uploaded
        if 'audio' not in request.files:
            return jsonify({
                "error": "No audio file provided"
            }), 400
            
        audio_file = request.files['audio']
        
        # Save the uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, 'uploaded_audio' + os.path.splitext(audio_file.filename)[1])
        audio_file.save(temp_path)
        
        # Convert audio format if needed
        converted_path = convert_audio_format(temp_path)
        
        # Preprocess the audio file
        processed_path = preprocess_audio(converted_path)
        
        # Extract features
        features = extract_features(processed_path)
        
        if features is None:
            return jsonify({
                "error": "Failed to extract features from the audio file"
            }), 400
            
        # Make prediction
        if USE_DUMMY_PREDICTION or model is None:
            # Dummy prediction for testing without a model
            emotion_idx = np.random.randint(0, len(EMOTIONS))
            emotion = EMOTIONS[emotion_idx]
            confidence = float(np.random.uniform(0.7, 0.95))
        else:
            # Reshape features for model input
            features = np.expand_dims(features, axis=0)
            
            # Predict emotion
            predictions = model.predict(features)[0]
            emotion_idx = np.argmax(predictions)
            emotion = EMOTIONS[emotion_idx]
            confidence = float(predictions[emotion_idx])
        
        # Clean up temporary files
        for file_path in [temp_path, converted_path, processed_path]:
            try:
                if file_path != temp_path and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing temporary file {file_path}: {e}")
        
        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "all_emotions": {emotion: float(conf) for emotion, conf in zip(EMOTIONS, predictions.tolist() if not USE_DUMMY_PREDICTION and model is not None else np.random.uniform(0, 0.3, len(EMOTIONS)).tolist())}
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/predict-live', methods=['POST'])
def predict_live():
    try:
        # Check if raw audio data was provided
        if 'audio_data' not in request.files:
            return jsonify({
                "error": "No audio data provided"
            }), 400
            
        audio_file = request.files['audio_data']
        
        # Save the uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, 'live_audio.wav')
        audio_file.save(temp_path)
        
        # Load audio data
        audio_data, sr = librosa.load(temp_path, sr=22050)
        
        # Extract features from live audio
        features = extract_features_live(audio_data, sr)
        
        if features is None:
            return jsonify({
                "error": "Failed to extract features from the live audio"
            }), 400
            
        # Make prediction
        if USE_DUMMY_PREDICTION or model is None:
            # Dummy prediction for testing without a model
            emotion_idx = np.random.randint(0, len(EMOTIONS))
            emotion = EMOTIONS[emotion_idx]
            confidence = float(np.random.uniform(0.7, 0.95))
            all_emotions = {e: float(np.random.uniform(0, 0.3)) for e in EMOTIONS}
            all_emotions[emotion] = confidence
        else:
            # Reshape features for model input
            features = np.expand_dims(features, axis=0)
            
            # Predict emotion
            predictions = model.predict(features)[0]
            emotion_idx = np.argmax(predictions)
            emotion = EMOTIONS[emotion_idx]
            confidence = float(predictions[emotion_idx])
            all_emotions = {emotion: float(conf) for emotion, conf in zip(EMOTIONS, predictions.tolist())}
        
        # Clean up temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"Error removing temporary file {temp_path}: {e}")
        
        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "all_emotions": all_emotions
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/get-emotions', methods=['GET'])
def get_emotions():
    """Return available emotion categories"""
    return jsonify({
        "emotions": EMOTIONS
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)