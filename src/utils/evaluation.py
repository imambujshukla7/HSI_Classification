from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import torch
import numpy as np

def evaluate_model(model, X_test, y_test, adjacency_matrix):
    """
    Evaluate the model's performance on the test data.

    Parameters:
    model (torch.nn.Module): The trained model.
    X_test (np.ndarray): Test data features.
    y_test (np.ndarray): True labels for the test data.
    adjacency_matrix (torch.Tensor): Hypergraph adjacency matrix for the test data.

    Returns:
    tuple: Overall Accuracy (OA), Average Accuracy (AA), and Kappa Accuracy.
    """
    model.eval()
    with torch.no_grad():
        # Convert test data to PyTorch tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Forward pass
        outputs = model(X_test_tensor, adjacency_matrix)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.numpy()
        y_test = np.array(y_test)
        
        # Calculate evaluation metrics
        oa = accuracy_score(y_test, predicted)
        kappa = cohen_kappa_score(y_test, predicted)
        report = classification_report(y_test, predicted, output_dict=True)
        aa = np.mean([report[str(i)]['recall'] for i in range(len(report) - 3)])  # -3 to exclude 'accuracy', 'macro avg', and 'weighted avg'

        print(f'Overall Accuracy (OA): {oa}')
        print(f'Average Accuracy (AA): {aa}')
        print(f'Kappa Accuracy: {kappa}')

        return oa, aa, kappa

# Test function to ensure correctness
if __name__ == "__main__":
    class DummyModel(torch.nn.Module):
        def forward(self, x, adj):
            return torch.rand(x.size(0), 10)  # Assume 10 classes for testing

    # Dummy data for testing
    model = DummyModel()
    X_test = np.random.rand(100, 30)  # Example test data with 100 samples, each with 30 features
    y_test = np.random.randint(0, 10, 100)  # Example test labels with 10 classes
    adjacency_matrix = torch.eye(100)  # Example adjacency matrix

    # Evaluate dummy model
    oa, aa, kappa = evaluate_model(model, X_test, y_test, adjacency_matrix)
    print(f'Test Evaluation - OA: {oa}, AA: {aa}, Kappa: {kappa}')
