import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from sklearn.metrics import euclidean_distances


def select_features(image_rgb):

    # Convert to grayscale for brightness and edge detection
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # --- Bright/Dark Detection ---
    # Calculate the local average using a 5x5 kernel
    local_mean = uniform_filter(gray.astype(np.float32), size=5)
    brightness_mask = gray > local_mean  # True if brighter than neighbors

    # --- Soft/Hard Detection ---
    # Use Laplacian to detect edges (higher value = more edge = "hard")
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_magnitude = np.abs(laplacian)
    edge_threshold = np.percentile(laplacian_magnitude, 75)  # upper quartile = hard
    hardness_mask = laplacian_magnitude > edge_threshold  # True = hard, False = soft

    # Visualize soft/hard and bright/dark
    soft_hard_map = np.zeros_like(gray)
    soft_hard_map[hardness_mask] = 255  # white = hard, black = soft

    bright_dark_map = np.zeros_like(gray)
    bright_dark_map[brightness_mask] = 255  # white = bright, black = dark

    # Display the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(soft_hard_map, cmap='gray')
    plt.title("Hard vs Soft (White = Hard)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(bright_dark_map, cmap='gray')
    plt.title("Bright vs Dark (White = Bright)")
    plt.axis("off")

    # Save the figure BEFORE showing it
    plt.savefig("clustering_comparison_Features.png", bbox_inches='tight')

    plt.tight_layout()
    #plt.show()

    return local_mean, laplacian_magnitude


def select_features(image_rgb):

    # Convert to grayscale for brightness and edge detection
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # --- Bright/Dark Detection ---
    # Calculate the local average using a 5x5 kernel
    local_mean = uniform_filter(gray.astype(np.float32), size=5)

    # --- Soft/Hard Detection ---
    # Slight blur before edge detection
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Laplacian to detect edges (higher value = more edge = "hard")
    laplacian = cv2.Laplacian(blurred_gray, cv2.CV_64F)
    edge_magnitude = np.abs(laplacian)

    # final smoothing to be consistent with the bright/dark detection window
    smoothed_edge_magnitude = uniform_filter(edge_magnitude, size=5)

    return local_mean, smoothed_edge_magnitude

def select_features_windowed(image_rgb, window_size):

    # Convert to grayscale for brightness and edge detection
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # --- Bright/Dark Detection ---
    # Calculate the local average using a 5x5 kernel
    local_mean = uniform_filter(gray.astype(np.float32), size=window_size)

    # --- Soft/Hard Detection ---
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Slight blur before edge detection
    blurred_gray = cv2.GaussianBlur(gray, (window_size, window_size), 0)

    # Use Laplacian to detect edges (higher value = more edge = "hard")
    laplacian = cv2.Laplacian(blurred_gray, cv2.CV_64F)
    edge_magnitude = np.abs(laplacian)

    # final smoothing to be consistent with the bright/dark detection window
    #smoothed_edge_magnitude = uniform_filter(edge_magnitude, size=window_size)

    return local_mean, edge_magnitude

def define_masks(local_mean, laplacian_magnitude, image_rgb):

    # Convert to grayscale for brightness and edge detection
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Generate the masks for improve visualization:

    # Mask for brightness
    brightness_mask = gray > local_mean  # True if brighter than neighbors
    # Mask for softness
    edge_threshold = np.percentile(laplacian_magnitude, 75)  # upper quartile = hard
    hardness_mask = laplacian_magnitude > edge_threshold  # True = hard, False = soft

    # Visualize soft/hard and bright/dark
    soft_hard_map = np.zeros_like(gray)
    soft_hard_map[hardness_mask] = 255  # white = hard, black = soft

    bright_dark_map = np.zeros_like(gray)
    bright_dark_map[brightness_mask] = 255  # white = bright, black = dark

    return soft_hard_map, bright_dark_map

def select_color_distance(mask, image_rgb):
    # --- Define the cozy cat bed sample area ---

    cat_bed_region = image_rgb[mask]
    cat_bed_pixels = cat_bed_region.reshape(-1, 3)
    mean_cat_color = np.mean(cat_bed_pixels, axis=0).reshape(1, -1)

    # --- Feature 3: Color Distance to Cat Bed ---
    all_pixels = image_rgb.reshape(-1, 3)
    color_distance = euclidean_distances(all_pixels, mean_cat_color)
    color_distance_scaled = (255 - color_distance).reshape(-1, 1)  # flip so closer is higher

    return color_distance_scaled

def select_softness_std(image_rgb, window_size):
    # Local texture variation
    from scipy.ndimage import generic_filter

    # Function to compute local standard deviation (texture)
    def local_std(arr, size=window_size):
        return generic_filter(arr, np.std, size=size)

    # Convert to grayscale for brightness and edge detection
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Compute local standard deviation (texture) in the grayscale image
    softness_std = local_std(gray, size=5)  # Change window size as needed

    return softness_std