import torch
import torch.nn as nn
import os

# Define a PyTorch LSTM model equivalent to the TensorFlow one
class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=40,  # Same as your feature dimension
            hidden_size=128,
            batch_first=True,
            return_sequences=True
        )
        self.dropout1 = nn.Dropout(0.4)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(0.3)
        
        # Dense layers
        self.dense1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(32, 6)  # 6 emotions
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        # Dense layers
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        x = self.softmax(x)
        
        return x

# Create the model
model = EmotionModel()

# Create a sample input for tracing
sample_input = torch.randn(1, 174, 40)  # Batch size 1, 174 time steps, 40 features

# Convert to TorchScript via tracing
traced_model = torch.jit.trace(model, sample_input)

# Save the model
os.makedirs('models', exist_ok=True)
torch.jit.save(traced_model, 'models/emotion_model.pt')

print("PyTorch placeholder model created and saved to models/emotion_model.pt")