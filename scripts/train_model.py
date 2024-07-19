import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.models.hybrid_hypergraph_attention import build_model, create_hypergraph_adjacency_matrix
from src.utils.data_loader import load_data
from src.utils.preprocessing import preprocess_data
from src.utils.evaluation import evaluate_model

def train_model(data_path, model_save_path, n_components=30, n_segments=100, num_epochs=100, lr=0.001, test_size=0.2, random_state=42):
    """
    Train the Hybrid Hypergraph Attention Network model.

    Parameters:
    data_path (str): Path to the dataset file.
    model_save_path (str): Path to save the trained model.
    n_components (int): Number of PCA components.
    n_segments (int): Number of superpixels for SLIC.
    num_epochs (int): Number of training epochs.
    lr (float): Learning rate for the optimizer.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Seed used by the random number generator.
    """
    try:
        # Load dataset
        print("Loading dataset...")
        data, labels = load_data(data_path)
        print("Dataset loaded. Data shape:", data.shape, "Labels shape:", labels.shape)

        # Preprocess data
        print("Preprocessing data...")
        segments = preprocess_data(data, n_components=n_components, n_segments=n_segments)
        print("Data preprocessed. Segments shape:", segments.shape)

        # Split the dataset into training and testing sets
        print("Splitting dataset into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=test_size, random_state=random_state)
        print("Dataset split. Training data shape:", X_train.shape, "Testing data shape:", X_test.shape)

        # Create hypergraph adjacency matrix for training data
        print("Creating hypergraph adjacency matrix for training data...")
        adjacency_matrix_train = create_hypergraph_adjacency_matrix(X_train, n_superpixels=n_segments)
        print("Adjacency matrix created for training data. Shape:", adjacency_matrix_train.shape)

        # Build model
        print("Building the model...")
        model = build_model(input_dim=n_components, hidden_dim=64, output_dim=len(np.unique(labels)))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        print("Model built.")

        # Train model
        print("Training the model...")
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(torch.tensor(X_train, dtype=torch.float32), adjacency_matrix_train)
            loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        # Save the trained model
        torch.save(model.state_dict(), model_save_path)
        print(f"Model trained and saved as '{model_save_path}'.")

    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    # Ensure Git LFS is initialized and tracking .mat files
    # Run the following commands in your terminal:
    # cd HSI_Classification
    # git lfs install
    # git lfs track "*.mat"

    data_path = 'data/IP/IP.mat'
    model_save_path = 'model.pth'
    train_model(data_path, model_save_path)
