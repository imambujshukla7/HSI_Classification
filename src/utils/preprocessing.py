from sklearn.decomposition import PCA
from skimage.segmentation import slic
import numpy as np

def apply_pca(X, n_components=30):
    """
    Apply Principal Component Analysis (PCA) to reduce the dimensionality of the data.

    Parameters:
    X (np.ndarray): Input data array with shape (height, width, bands).
    n_components (int): Number of principal components to keep.

    Returns:
    np.ndarray: Data array with reduced dimensions.
    """
    try:
        X_flat = X.reshape(-1, X.shape[-1])
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_flat)
        X_reduced = X_reduced.reshape(X.shape[0], X.shape[1], n_components)
    except Exception as e:
        raise RuntimeError(f"Error applying PCA: {e}")
    return X_reduced

def segment_image(X, n_segments=100):
    """
    Apply Simple Linear Iterative Clustering (SLIC) to segment the image into superpixels.

    Parameters:
    X (np.ndarray): Input data array with shape (height, width, bands).
    n_segments (int): Number of superpixels.

    Returns:
    np.ndarray: Segmented image array.
    """
    try:
        segments = slic(X, n_segments=n_segments, compactness=10)
    except Exception as e:
        raise RuntimeError(f"Error segmenting image: {e}")
    return segments

def preprocess_data(data, n_components=30, n_segments=100):
    """
    Preprocess the data by normalizing, applying PCA, and segmenting into superpixels.

    Parameters:
    data (np.ndarray): Input data array with shape (height, width, bands).
    n_components (int): Number of principal components to keep.
    n_segments (int): Number of superpixels.

    Returns:
    np.ndarray: Preprocessed data array.
    """
    data_normalized = normalize_data(data)
    data_pca = apply_pca(data_normalized, n_components)
    segments = segment_image(data_pca, n_segments)
    return segments

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

# Test functions to ensure correctness
if __name__ == "__main__":
    # Dummy data for testing
    data = np.random.rand(100, 100, 200)  # Example hyperspectral data with 200 bands
    
    # Normalize data
    normalized_data = normalize_data(data)
    print("Normalized data range:", np.min(normalized_data), np.max(normalized_data))
    
    # Apply PCA
    data_pca = apply_pca(normalized_data, n_components=30)
    print("PCA applied. New shape:", data_pca.shape)
    
    # Segment image
    segments = segment_image(data_pca, n_segments=100)
    print("Image segmented. Segments shape:", segments.shape)
    
    # Preprocess data
    preprocessed_data = preprocess_data(data, n_components=30, n_segments=100)
    print("Preprocessed data shape:", preprocessed_data.shape)
