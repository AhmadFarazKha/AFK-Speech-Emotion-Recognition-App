import librosa
import numpy as np

def extract_features(audio_path, max_pad_len=174):
    """
    Extract audio features from an audio file for speech emotion recognition
    
    Args:
        audio_path: Path to audio file
        max_pad_len: Length to pad/truncate MFCC features
        
    Returns:
        features: Numpy array of audio features
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Normalize MFCC
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        
        # Transpose to get time as first dimension
        mfccs = mfccs.T
        
        # Pad or truncate MFCC to fit the fixed length
        if mfccs.shape[0] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfccs = mfccs[:max_pad_len, :]
            
        return mfccs
    
    except Exception as e:
        print(f"Error encountered while parsing file: {e}")
        return None

def extract_features_live(audio_data, sr, max_pad_len=174):
    """
    Extract audio features from live audio data for speech emotion recognition
    
    Args:
        audio_data: Raw audio data
        sr: Sample rate
        max_pad_len: Length to pad/truncate MFCC features
        
    Returns:
        features: Numpy array of audio features
    """
    try:
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        
        # Extract additional features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        
        # Normalize features
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-10)
        chroma = (chroma - np.mean(chroma)) / (np.std(chroma) + 1e-10)
        mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-10)
        spectral_contrast = (spectral_contrast - np.mean(spectral_contrast)) / (np.std(spectral_contrast) + 1e-10)
        
        # Transpose to get time as first dimension
        mfccs = mfccs.T
        
        # Pad or truncate MFCC to fit the fixed length
        if mfccs.shape[0] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfccs = mfccs[:max_pad_len, :]
            
        return mfccs
    
    except Exception as e:
        print(f"Error encountered while extracting features: {e}")
        return None