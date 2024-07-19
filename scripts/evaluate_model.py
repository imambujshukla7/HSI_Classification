import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.models.hybrid_hypergraph_attention import build_model, create_hypergraph_adjacency_matrix
from src.utils.data_loader import load_data
from src.utils.preprocessing import preprocess_data
from src.utils.evaluation import evaluate_model

def evaluate_model_script(data_path, model_path, n_components=30, n_segments=100, test_size=0.2, random_state=42):
    """
    Evaluate the trained Hybrid Hypergraph Attention Network model.

    Parameters:
    data_path (str): Path to the dataset file.
    model_path (str): Path to the trained model file.
    n_components (int): Number of PCA components.
    n_segments (int): Number of superpixels for SLIC.
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

        # Create hypergraph adjacency matrix for testing data
        print("Creating hypergraph adjacency matrix for testing data...")
        adjacency_matrix_test = create_hypergraph_adjacency_matrix(X_test, n_superpixels=n_segments)
        print("Adjacency matrix created for testing data. Shape:", adjacency_matrix_test.shape)

        # Build model
        print("Building the model...")
        model = build_model(input_dim=n_components, hidden_dim=64, output_dim=len(np.unique(labels)))
        model.load_state_dict(torch.load(model_path))
        print("Model built and loaded with trained weights.")

        # Evaluate model
        print("Evaluating the model...")
        oa, aa, kappa = evaluate_model(model, X_test, y_test, adjacency_matrix_test)
        print(f'Evaluation results - OA: {oa}, AA: {aa}, Kappa: {kappa}')

    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    data_path = 'data/IP/IP.mat'
    model_path = 'model.pth'
    evaluate_model_script(data_path, model_path)
