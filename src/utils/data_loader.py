import h5py
import numpy as np

def load_data(file_path):
    """
    Load data and labels from an HDF5 file.

    Parameters:
    file_path (str): Path to the .mat file.

    Returns:
    data (np.ndarray): Loaded data array.
    labels (np.ndarray): Loaded labels array.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Assuming the data is stored in the 'data' key and labels in the 'labels' key
            data = np.array(f['data'])
            labels = np.array(f['labels'])
    except KeyError as e:
        raise KeyError(f"Missing key in HDF5 file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading data from {file_path}: {e}")
    
    return data, labels

def normalize_data(data):
    """
    Normalize data to [0, 1] range.

    Parameters:
    data (np.ndarray): Input data array.

    Returns:
    np.ndarray: Normalized data array.
    """
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min == 0:
        raise ValueError("Data has no variation.")
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data

def split_train_test(data, labels, test_size=0.2, random_state=None):
    """
    Split data into training and testing sets.

    Parameters:
    data (np.ndarray): Input data array.
    labels (np.ndarray): Input labels array.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    X_train, X_test, y_train, y_test (tuple): Split training and testing data and labels.
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Test functions to ensure correctness
if __name__ == "__main__":
    # Assuming the 'data' key and 'labels' key exist in the test HDF5 file
    test_file_path = 'data/IP/IP.mat'

    # Load data
    data, labels = load_data(test_file_path)
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)

    # Normalize data
    normalized_data = normalize_data(data)
    print("Normalized data range:", np.min(normalized_data), np.max(normalized_data))

    # Split data
    X_train, X_test, y_train, y_test = split_train_test(normalized_data, labels)
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing labels shape:", y_test.shape)
