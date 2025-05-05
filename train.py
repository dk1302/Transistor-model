import cnn
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn

def train(model_path, params_path, features_path, epochs=170):

    dataset = cnn.CoordinatesDataset(
        parameters_file=params_path,
        coordinates_file=features_path
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    if __name__ == "__main__":
        # Initialize your model, optimizer, criterion
        model = cnn.CNN(input=1, output=101)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
        criterion = nn.MSELoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create trainer
        trainer = cnn.ModelTrainer(
            model=model.to(device),
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        
        # Train and evaluate
        trainer.train(train_loader, val_loader, epochs)
        
        # Plot training history
        trainer.plot_history()

        torch.save(model, model_path)

train('cnn_model_v1.pth', 'new_params.csv', 'new_features_v1.csv')
train('cnn_model_v5.pth', 'new_params.csv', 'new_features_v5.csv')
train('cnn_model_v10.pth', 'new_params.csv', 'new_features_v10.csv')

