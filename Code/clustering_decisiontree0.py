import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch, Ellipse
from scipy.ndimage import uniform_filter, binary_dilation
from sklearn.cluster import KMeans
from PIL import Image

# Load the image
from sklearn.metrics import euclidean_distances
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

import in_out_methods as io
import sample_choice as sc
import clustering_w_sample as cl_sample
import select_features as sf

def clustering_and_tree(ellipse_mask, mask_type, selected_k, max_iter, image_rgb):
    print("################################################")
    print("K-means an the decision tree with 2 labels (Cat Nest, Wall)")
    print("A subset of the images pixels are used for the calculation of the tree. Pixels of the cluster including most pixels of the cat nest are used.")

    # Repeat the clustering
    labels_masked2, segmented_image_masked2 = cl_sample.cluster_w_mask2(ellipse_mask, mask_type, selected_k, max_iter, image_rgb)

    # Calculate the features
    brightness, softness = sf.select_features_windowed(image_rgb, 3)
    soft_hard_map, bright_dark_map = sf.define_masks(brightness, softness, image_rgb)

    # Reshape to use as feature
    brightness = brightness.reshape(-1, 1)
    softness = softness.reshape(-1, 1)
    features_all = np.hstack([softness, brightness])

    # Generate all necessary masks (Catbed and Clusterlabels):

    # Catbed: the area including the catbed
    # Create an empty mask
    catbed_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    catbed_mask[ellipse_mask] = True

    # Flatten both masks
    catbed_mask_flat = catbed_mask.flatten()
    labels_masked2_flat = labels_masked2.flatten()

    # Only consider pixels inside the cat-bed mask
    labels_in_catbed = labels_masked2_flat[catbed_mask_flat]

    # Count how often each cluster label appears inside the cat-bed
    from collections import Counter
    cluster_counts = Counter(labels_in_catbed)

    # Get the cluster with the most cat-bed pixels
    best_cluster_label = cluster_counts.most_common(1)[0][0]
    print(f"Cluster most aligned with cat-bed: {best_cluster_label}")

    cluster_mask = (labels_masked2 == best_cluster_label)  # shape: (H, W)
    cluster_mask_flat = cluster_mask.flatten()

    # Filter features to just cluster 1
    X = features_all[cluster_mask_flat]

    # Binary labels for catbed - cluster pixels:
    # 1 = in cat-bed, 0 = in cluster 1 but not cat-bed
    y = catbed_mask_flat[cluster_mask_flat].astype(int)

    print("Check for cat-bed existence: ")
    print(np.unique(y, return_counts=True))
    print("Check if label correct: ")
    print(np.unique(labels_in_catbed, return_counts=True))
    print(best_cluster_label)

    # Train the tree
    tree = DecisionTreeClassifier(class_weight="balanced", max_depth=3,min_samples_split=10, random_state=42)
    tree.fit(X, y)

    print("Class distribution:", np.unique(y, return_counts=True))

    # Predict only on the masked pixels
    tree_predictions_masked = tree.predict(X)

    # Create a full-size blank array and fill only the masked locations
    tree_predictions = np.full(image_rgb.shape[:2], fill_value=-1, dtype=int)

    # Fill in the predictions for the masked pixels
    tree_predictions_flat = tree_predictions.flatten()
    tree_predictions_flat[cluster_mask_flat] = tree_predictions_masked

    # Reshape the predictions back to the original image shape
    tree_predictions = tree_predictions_flat.reshape(image_rgb.shape[:2])

    # Visualize the decision tree predictions
    plt.imshow(tree_predictions, cmap='tab10')
    plt.title("Decision Tree Predictions on Cat Nest Cluster")
    plt.axis("off")
    plt.show()

    # Visualize the tree
    plt.figure(figsize=(8, 6))
    plot_tree(tree, feature_names=["softness", "brightness"], class_names=["Other", "Cat-Bed"], filled=True)
    plt.title("Refinement Tree for the Cat Nest Cluster")
    plt.show()

    print(export_text(tree, feature_names=["softness", "brightness"]))

    # Access the tree's structure
    tree_ = tree.tree_
    feature_names = ["brightness", "softness"]

    print("Decision Tree Splits:\n")
    for i in range(tree_.node_count):
        if tree_.children_left[i] != tree_.children_right[i]:  # It's a split node
            feature = feature_names[tree_.feature[i]]
            threshold = tree_.threshold[i]
            print(f"Node {i}: if {feature} <= {threshold:.4f}")


    # Predefined color palette for up to 8 decision nodes
    color_palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
    ]

    # Initialize overlay with all black (or transparent)
    overlay_image = np.zeros_like(image_rgb)

    # Get unique nodes from within the target cluster only
    mask_in_cluster = (labels_masked2 == best_cluster_label)
    tree_preds_in_cluster = tree_predictions[mask_in_cluster]
    unique_nodes = np.unique(tree_preds_in_cluster)

    for node in np.unique(tree_predictions):
        mask_tmp = (tree_predictions == node) & (labels_masked2 ==  best_cluster_label)
        print(f"Node {node}: {np.count_nonzero(mask_tmp)} pixels in cluster")

    # Step 7: Dynamically create a color palette based on the number of unique nodes
    unique_nodes = np.unique(tree_predictions)
    num_nodes = len(unique_nodes)

    node_classes = tree.tree_.value  # This is a (n_nodes, n_classes) array
    class_names = ["Cat-Bed", "Other"]  # Assuming binary classification: 0 for Other, 1 for Cat-Bed

    # Step 3: Print out the class for each node
    for node_id in range(node_classes.shape[0]):
        # Find the class with the maximum value in each node's value (distribution)
        class_index = np.argmax(node_classes[node_id])  # Index of the class with the highest count

        # Get the class name based on the index (assuming binary: 0 -> "Other", 1 -> "Cat-Bed")
        node_class = class_names[class_index] # Boolean indexing

        # Print the node ID and the most common class in that node
        print(f"Node {node_id}: Class = {node_class}")


    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#17becf', '#8c564b', '#e377c2', '#bcbd22', '#7f7f7f']

    legend_handles = []
    # Apply colors only inside the target cluster
    for i, node_label in enumerate(unique_nodes):
        if i >= len(color_palette):
            print(f"Warning: more nodes than colors. Node {node_label} skipped.")
            continue
        # Convert hex to float RGB and int RGB
        hex_color = colors[i]
        rgb_float = matplotlib.colors.to_rgb(hex_color)  # (0.12, 0.47, 0.71)
        rgb_255 = np.array([int(c * 255) for c in rgb_float])  # (31, 119, 180)

        # Create a mask for the current node within the cluster
        node_mask = (tree_predictions == node_label) & (labels_masked2 ==  best_cluster_label) #& (labels == target_cluster_id)

        # Apply color to those pixels
        overlay_image[node_mask] = rgb_255

        #Generate an informative legend
        class_index = np.argmax(node_classes[node_label])  # Index of the class with the highest count
        node_class = class_names[class_index]
        legend_handles.append(Patch(color=rgb_float, label=f'Node {node_label}: Class = {node_class}'))
        print("Size overlay: ")
        print(i)
        print(overlay_image.shape)

    plt.imshow(overlay_image)
    plt.title("Overlay Image Only")
    plt.axis("off")
    #plt.show()

    # Plot the overlay on top of the original image
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.imshow(overlay_image, alpha=0.75)
    plt.title(f"Tree Nodes in Cluster { best_cluster_label}")

    # Add the legend to the plot
    plt.legend(handles=legend_handles, loc='upper left', fontsize=12, title="Nodes")
    plt.axis("off")
    plt.show()