import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split

from models.hybrid_hypergraph_attention import build_model, create_hypergraph_adjacency_matrix
from utils.data_loader import load_data
from utils.preprocessing import preprocess_data
from utils.evaluation import evaluate_model

def main():
    # Load dataset
    print("Loading dataset...")
    data, labels = load_data('data/IP/IP.mat')
    print("Dataset loaded. Data shape:", data.shape, "Labels shape:", labels.shape)

    # Preprocess data
    print("Preprocessing data...")
    segments = preprocess_data(data, n_components=30, n_segments=100)
    print("Data preprocessed. Segments shape:", segments.shape)

    # Split the dataset into training and testing sets
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, random_state=42)
    print("Dataset split. Training data shape:", X_train.shape, "Testing data shape:", X_test.shape)

    # Create hypergraph adjacency matrix for training data
    print("Creating hypergraph adjacency matrix for training data...")
    adjacency_matrix_train = create_hypergraph_adjacency_matrix(X_train, n_superpixels=100)
    print("Adjacency matrix created for training data. Shape:", adjacency_matrix_train.shape)

    # Create hypergraph adjacency matrix for testing data
    print("Creating hypergraph adjacency matrix for testing data...")
    adjacency_matrix_test = create_hypergraph_adjacency_matrix(X_test, n_superpixels=100)
    print("Adjacency matrix created for testing data. Shape:", adjacency_matrix_test.shape)

    # Build model
    print("Building the model...")
    model = build_model(input_dim=30, hidden_dim=64, output_dim=len(np.unique(labels)))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    print("Model built.")

    # Train model
    print("Training the model...")
    model.train()
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32), adjacency_matrix_train)
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    print("Model trained.")

    # Evaluate model
    print("Evaluating the model...")
    oa, aa, kappa = evaluate_model(model, X_test, y_test, adjacency_matrix_test)
    print(f'Evaluation results - OA: {oa}, AA: {aa}, Kappa: {kappa}')

if __name__ == "__main__":
    main()
