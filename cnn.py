import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d

# Set random seed for reproducibility
torch.manual_seed(42)

## 1. Custom Dataset Class for 102x102 Coordinates
class CoordinatesDataset(Dataset):
    def __init__(self, parameters_file, coordinates_file):
        # Load parameters
        self.parameters = pd.read_csv(parameters_file).values.astype(np.float32)
        
        # Load coordinates and separate x and y
        self.coords = pd.read_csv(coordinates_file).values.astype(np.float32)  # First 102 columns are x
        
        # Normalize parameters
        self.param_scaler = MinMaxScaler()
        self.parameters = self.param_scaler.fit_transform(self.parameters)
        
        # Normalize coordinates separately
        self.coords_scaler = MinMaxScaler()
        self.coords = self.coords_scaler.fit_transform(self.coords)
        
    def __len__(self):
        return len(self.parameters)
    
    def __getitem__(self, idx):
        params = torch.tensor(self.parameters[idx], dtype=torch.float32)
        coords = torch.tensor(self.coords[idx], dtype=torch.float32)
        return params, coords  
    
    def inverse_transform(self, coordinates):
        """Convert normalized coordinates back to original scale"""
        return self.coords_scaler.inverse_transform(coordinates)

dataset_v1 = CoordinatesDataset(
    parameters_file='datasets/new_params.csv',
    coordinates_file='datasets/new_features_v1.csv'
)

dataset_v5 = CoordinatesDataset(
    parameters_file='datasets/new_params.csv',
    coordinates_file='datasets/new_features_v5.csv'
)

dataset_v10 = CoordinatesDataset(
    parameters_file='datasets/new_params.csv',
    coordinates_file='datasets/new_features_v10.csv'
)

scale_v1_dataset = CoordinatesDataset(
    parameters_file='datasets/new_params.csv',
    coordinates_file='datasets/scale_v1.csv'
)
scale_v5_dataset = CoordinatesDataset(
    parameters_file='datasets/new_params.csv',
    coordinates_file='datasets/scale_v5.csv'
)
scale_v10_dataset = CoordinatesDataset(
    parameters_file='datasets/new_params.csv',
    coordinates_file='datasets/scale_v10.csv'
)

class ResidualBlock(nn.Module):
    """Residual block with additional hidden layer"""
    def __init__(self, in_channels, out_channels, stride=1, 
                 dropout_rate=0.2, use_batchnorm=True):
        super(ResidualBlock, self).__init__()
        
        # Main convolutional path with additional hidden layer
        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels//2) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Additional hidden layer
            nn.Conv1d(out_channels//2, out_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels//2) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            layers = [
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            ]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_channels))
            self.shortcut = nn.Sequential(*layers)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_path(x)
        out += residual
        out = self.relu(out)
        return out


class SelfAttention(nn.Module):
    """Self-attention mechanism"""
    def __init__(self, channel_size):
        super(SelfAttention, self).__init__()
        
        self.query = nn.Conv1d(channel_size, channel_size // 8, 1)
        self.key = nn.Conv1d(channel_size, channel_size // 8, 1)
        self.value = nn.Conv1d(channel_size, channel_size, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, C, width = x.size()
        
        # Calculate queries, keys, values
        q = self.query(x).view(batch_size, -1, width).permute(0, 2, 1)  # (B, W, C')
        k = self.key(x).view(batch_size, -1, width)  # (B, C', W)
        v = self.value(x).view(batch_size, -1, width)  # (B, C, W)
        
        # Calculate attention
        attention = torch.bmm(q, k)  # (B, W, W)
        attention = self.softmax(attention)
        
        # Apply attention to values
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width)
        
        # Add skip connection
        return self.gamma * out + x

class CNN(nn.Module):
    def __init__(self, input = 1, output = 202, dropout_rate=0.5, use_batchnorm=True, 
                 use_residual=True, use_attention=True, 
                 hidden_channels=128):
        super(CNN, self).__init__()
        
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.use_batchnorm = use_batchnorm
        
        # Initial convolutional block with additional hidden layer
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(64, hidden_channels, kernel_size=5, stride=1, padding=2),  # New hidden layer
            nn.BatchNorm1d(hidden_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks with additional hidden layers
        self.res_block1 = ResidualBlock(hidden_channels, 256, stride=2, 
                                             dropout_rate=dropout_rate,
                                             use_batchnorm=use_batchnorm)
        
        self.res_block2 = ResidualBlock(256, 512, stride=2,
                                              dropout_rate=dropout_rate,
                                              use_batchnorm=use_batchnorm)
        
        # Attention mechanism
        if use_attention:
            self.attention = SelfAttention(512)
        
        # Additional convolutional blocks with more hidden layers
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1),  # New hidden layer
            nn.BatchNorm1d(1024) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1024) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Enhanced classifier with additional hidden layer
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),  # New hidden layer
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output)
        )
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
    
        # Initial block
        x = self.conv_block1(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Attention
        if self.use_attention:
            x = self.attention(x)
        
        # Final conv block
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        return x

class ModelTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.reset_history()
        
    def reset_history(self):
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def training(self, dataloader, device='cpu'):
        self.model.train()  # Set model to training mode
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()    # Scale the gradients
            self.optimizer.step()          # Update the model parameters

            running_loss += loss.item() * inputs.size(0)

        # Calculate the training loss and training accuracy
        train_loss = running_loss / len(dataloader.dataset)
        self.history['train_loss'].append(train_loss)
        
        return train_loss

    def evaluate(self, dataloader):
        self.model.eval()  # Set model to evaluation mode
        # evaluate on the validation set
        val_loss = 0.0
        with torch.no_grad():
            for data in dataloader:
                params, labels = data
                params = params.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(params)

                val_loss += self.criterion(outputs, labels).item() * labels.size(0)

        # Calculate the validation accuracy and validation loss
        val_loss /= len(dataloader.dataset)
        self.history['val_loss'].append(val_loss)
        
        return val_loss
    
    
    def train(self, train_loader, val_loader, epochs):
        self.reset_history()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self.training(train_loader)
            val_loss = self.evaluate(val_loader)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def plot_history(self):
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.show()

class use_model:
    def __init__(self, params_path, index=0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.index = index

        self.input_data = pd.read_csv(params_path).values.astype(np.float32)

        self.x_axis = np.empty(97)
        self.x_axis[1] = 0.5
        for x in range(95):
            self.x_axis[x+2] = self.x_axis[x+1]+0.1
    
    def run(self, model_path, scale_path, v=1):
        # Load dataset based on CG1 Voltage
        if(v == 1):
            dataset = dataset_v1
            scale_dataset = scale_v1_dataset
        elif(v == 5):
            dataset = dataset_v5
            scale_dataset = scale_v5_dataset
        elif(v == 10):
            dataset = dataset_v10
            scale_dataset = scale_v10_dataset

        input_data = dataset.param_scaler.transform(self.input_data)

        sample = input_data[self.index]
        sample = sample[np.newaxis, :]

        model = torch.load(model_path, weights_only=False)
        model.eval()  # Set model to evaluation mode
        model.to(self.device)
        scale = torch.load(scale_path, weights_only=False)
        scale.eval()  # Set model to evaluation mode
        scale.to(self.device)

        # Ensure input is a tensor and add batch dimension if needed
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)

        if sample.dim() == 2:  # If single sample (channels, length)
            sample = sample.unsqueeze(0)  # Add batch dimension

        sample = sample.to(self.device)

        with torch.no_grad():  # Disable gradient calculation
            outputs = model(sample)
            scaleValues = scale(sample)
        
        scaleValue = scaleValues.numpy()
        scaleValue = scale_dataset.inverse_transform(scaleValue)
        scaleValue = scaleValue.flatten()

        # Convert normalized coordinates back to original scale
        output = outputs.numpy()
        output = dataset.inverse_transform(output)
        output = output.flatten()

        item = str(output[0])
        reverse_scale = ''
        scale = False
        for char in item:
            if(char == '-'):
                scale=True
                continue
            if(scale):
                reverse_scale += char
        
        reverse_scale = int(reverse_scale)

        output = output*10**reverse_scale
        output = output*10**-scaleValue[0]
        output = np.insert(output, 0, 0.0)

        self.plot_prediction(self.x_axis, output)
        
    def plot_prediction(self, x_axis, outputs):
        plt.figure(figsize=(9, 5))

        #Makes graph smoother
        cubic_interpolation_model = interp1d(x_axis, outputs, kind = "cubic")

        x_axis = np.linspace(x_axis.min(), x_axis.max(), 9)

        y_axis = cubic_interpolation_model(x_axis)
                
        # Loss plot
        plt.plot(x_axis, y_axis)
        plt.title('Surrogate Model')
        plt.xlabel('Drain Voltage (V)')
        plt.ylabel('Drain Current (A)') 
        plt.show()
