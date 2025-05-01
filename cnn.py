import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
torch.manual_seed(42)

## 1. Custom Dataset Class for 102x102 Coordinates
class XYCoordinatesDataset(Dataset):
    def __init__(self, parameters_file, coordinates_file):
        # Load parameters
        self.parameters = pd.read_csv(parameters_file).values.astype(np.float32)
        
        # Load coordinates and separate x and y
        self.coords = pd.read_csv(coordinates_file).values.astype(np.float32)  # First 102 columns are x
        
        # Normalize parameters
        self.param_scaler = StandardScaler()
        self.parameters = self.param_scaler.fit_transform(self.parameters)
        
        # Normalize coordinates separately
        self.coords_scaler = StandardScaler()
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

## 2. Load Your Custom Dataset
dataset = XYCoordinatesDataset(
    parameters_file='params.csv',
    coordinates_file='features.csv',
)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class EnhancedResidualBlock(nn.Module):
    """Residual block with additional hidden layer"""
    def __init__(self, in_channels, out_channels, stride=1, 
                 dropout_rate=0.2, use_batchnorm=True):
        super(EnhancedResidualBlock, self).__init__()
        
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


class SelfAttention1D(nn.Module):
    """Self-attention mechanism for 1D data"""
    def __init__(self, channel_size):
        super(SelfAttention1D, self).__init__()
        
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

class ComplexCNN1D(nn.Module):
    def __init__(self, input = 1, output = 202, dropout_rate=0.5, use_batchnorm=True, 
                 use_residual=True, use_attention=True, 
                 hidden_channels=128):
        super(ComplexCNN1D, self).__init__()
        
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
        self.res_block1 = EnhancedResidualBlock(hidden_channels, 256, stride=2, 
                                             dropout_rate=dropout_rate,
                                             use_batchnorm=use_batchnorm)
        
        self.res_block2 = EnhancedResidualBlock(256, 512, stride=2,
                                              dropout_rate=dropout_rate,
                                              use_batchnorm=use_batchnorm)
        
        # Attention mechanism
        if use_attention:
            self.attention = SelfAttention1D(512)
        
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
            'val_loss': [],
        }
    
    def training(self, dataloader, device='cpu'):
        self.model.train()  # Set model to training mode
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()    # Scale the gradients
            optimizer.step()          # Update the model parameters

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
                params = params.to(device)
                labels = labels.to(device)

                outputs = model(params)
                val_loss += criterion(outputs, labels).item() * labels.size(0)

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


# Example usage:
if __name__ == "__main__":
    # Initialize your model, optimizer, criterion
    model = ComplexCNN1D(input=1, output=202)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create trainer
    trainer = ModelTrainer(
        model=model.to(device),
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
    
    # Train and evaluate
    trainer.train(train_loader, val_loader, epochs=250)
    
    # Plot training history
    trainer.plot_history()