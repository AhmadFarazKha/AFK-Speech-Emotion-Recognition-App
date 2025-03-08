import librosa
import numpy as np
import soundfile as sf
import tempfile
import os

def preprocess_audio(file_path):
    """
    Preprocess audio file for speech emotion recognition
    
    Args:
        file_path: Path to audio file
        
    Returns:
        processed_path: Path to processed audio file
    """
    try:
        # Create a temporary file for the processed audio
        temp_dir = tempfile.gettempdir()
        processed_path = os.path.join(temp_dir, 'processed_audio.wav')
        
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Save processed audio
        sf.write(processed_path, y, sr)
        
        return processed_path
    
    except Exception as e:
        print(f"Error in preprocessing audio: {e}")
        return file_path

def convert_audio_format(file_path):
    """
    Convert audio to compatible format for model if needed
    
    Args:
        file_path: Path to audio file
        
    Returns:
        converted_path: Path to converted audio file
    """
    try:
        # Get file extension
        _, file_extension = os.path.splitext(file_path)
        
        # If already WAV format, return original path
        if file_extension.lower() == '.wav':
            return file_path
        
        # Create a temporary file for the converted audio
        temp_dir = tempfile.gettempdir()
        converted_path = os.path.join(temp_dir, 'converted_audio.wav')
        
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)
        
        # Save as WAV
        sf.write(converted_path, y, sr)
        
        return converted_path
    
    except Exception as e:
        print(f"Error in converting audio format: {e}")
        return file_path