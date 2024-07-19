from skimage.segmentation import slic
import numpy as np

def segment_image(image, n_segments=100, compactness=10):
    """
    Apply Simple Linear Iterative Clustering (SLIC) to segment the image into superpixels.

    Parameters:
    image (np.ndarray): Input image array with shape (H, W, C).
    n_segments (int): Number of superpixels.
    compactness (int): Balances color proximity and space proximity. Higher values give more weight to space proximity.

    Returns:
    np.ndarray: Segmented image array with superpixel labels.
    """
    try:
        segments = slic(image, n_segments=n_segments, compactness=compactness)
    except Exception as e:
        raise RuntimeError(f"Error segmenting image: {e}")
    return segments

def create_superpixel_segments(data, n_segments=100, compactness=10):
    """
    Create superpixel segments for a batch of images.

    Parameters:
    data (np.ndarray): Input data array with shape (N, H, W, C), where N is the number of images.
    n_segments (int): Number of superpixels.
    compactness (int): Balances color proximity and space proximity. Higher values give more weight to space proximity.

    Returns:
    np.ndarray: Array of superpixel segments for each image.
    """
    if len(data.shape) != 4:
        raise ValueError("Input data must be a 4D array with shape (N, H, W, C).")

    superpixels = []
    for i in range(data.shape[0]):
        segments = segment_image(data[i], n_segments=n_segments, compactness=compactness)
        superpixels.append(segments)
    return np.array(superpixels)

# Test functions to ensure correctness
if __name__ == "__main__":
    # Dummy data for testing
    data = np.random.rand(5, 100, 100, 3)  # Example batch of 5 images with shape (100, 100, 3)

    # Segment a single image
    single_image = data[0]
    segments = segment_image(single_image, n_segments=100, compactness=10)
    print("Single image segmented. Segments shape:", segments.shape)

    # Create superpixel segments for the batch of images
    superpixel_segments = create_superpixel_segments(data, n_segments=100, compactness=10)
    print("Superpixel segments created for batch. Segments shape:", superpixel_segments.shape)
