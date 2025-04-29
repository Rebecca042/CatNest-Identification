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

def clustering_and_tree2(ellipse_mask, mask_type, selected_k, max_iter, image_rgb):
    print("################################################")
    print("K-means an the decision tree with 2 labels (Cat Nest, Wall)")
    print("All image pixels are used for the calculation of the tree")
    print("The additional feature 'texture' is introduced.")
    image_marked_shape = image_rgb_shape = image_rgb.shape
    # Repeat the clustering
    labels_masked3, segmented_image_masked3 = cl_sample.cluster_w_mask2(ellipse_mask, mask_type, selected_k, max_iter, image_rgb)
    # Display the Brightness and Softness along with the Original Image
    titles = ["xxxxxxxxxxx"]
    title = "Sample Area for Decision Tree - Cat Nest"
    io.prepare_image_overlay(segmented_image_masked3, ellipse_mask, title)
    # Calculate the features
    brightness, softness = sf.select_features_windowed(image_rgb, 3)
    soft_hard_map, bright_dark_map = sf.define_masks(brightness, softness, image_rgb)

    softness_std = sf.select_softness_std(image_rgb, 5)

    # Optionally, visualize the texture variation (this can help verify the feature extraction)
    import matplotlib.pyplot as plt
    plt.imshow(softness_std, cmap='hot')
    plt.colorbar()
    plt.title("Local Standard Deviation (Texture)")
    plt.show()

    # Reshape to use as feature
    brightness = brightness.reshape(-1, 1)
    softness = softness.reshape(-1, 1)
    # Flatten the result to make it suitable for feature input
    softness_std_flat = softness_std.reshape(-1,1)

    features_all = np.hstack([softness, softness_std_flat, brightness])

    # Generate all necessary masks (Catbed and Clusterlabels):

    # Catbed: the area including the catbed
    # Create an empty mask
    catbed_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    catbed_mask[ellipse_mask] = 1
    catbed_mask_flat = catbed_mask.flatten()

    # Only consider pixels inside the cat-bed mask
    labels_masked3_flat = labels_masked3.flatten()
    labels_in_catbed3 = labels_masked3_flat[catbed_mask_flat]
    # Count how often each cluster label appears inside the cat-bed
    from collections import Counter
    cluster_counts2 = Counter(labels_in_catbed3)

    # Get the cluster with the most cat-bed pixels
    best_cluster_label2 = cluster_counts2.most_common(1)[0][0]
    print(f"Cluster most aligned with cat-bed: {best_cluster_label2}")

    ## Select the couch as second most liked spot - assign 2
    start_x, end_x, start_y, end_y  = sc.select_area(image_marked_shape, 260, 680, 540,650)
    couch_sample = (start_y, end_y,start_x, end_x)
    mask_couch = sc.select_ellipse(couch_sample, image_rgb_shape)
    title = "Add Sample Area for Decision Tree - Couch"
    io.prepare_image_overlay(image_rgb, mask_couch, title)

    ## Select the wall, to have a definition of disliked spots - assign -1
    start_x, end_x, start_y, end_y  = sc.select_area(image_marked_shape, 150, 550, 50, 250)
    wall_sample = (start_y, end_y,start_x, end_x)
    mask_wall = sc.select_ellipse(wall_sample, image_rgb_shape)
    title = "Add Sample Area for Decision Tree - Wall"
    io.prepare_image_overlay(image_rgb, mask_wall, title)

    ## Select the table, the cat likes it - assign 2
    start_x, end_x, start_y, end_y  = sc.select_area(image_marked_shape, 270, 720, 650, 800)
    table_sample = (start_y, end_y,start_x, end_x)
    mask_table = sc.select_ellipse(table_sample, image_rgb_shape)
    title = "Add Sample Area for Decision Tree - Table"
    io.prepare_image_overlay(image_rgb, mask_table, title)

    cluster_mask2 = (labels_masked3 == best_cluster_label2)  # shape: (H, W)
    cluster_mask_flat2 = cluster_mask2.flatten()


    # Binary labels for catbed - cluster pixels:
    # 1 = in cat-bed, 0 = in cluster 1 but not cat-bed
    y_full = np.zeros(image_rgb.shape[:2], dtype=int).flatten()
    mask_flat = ellipse_mask.flatten()
    mask_couch_flat = mask_couch.flatten()
    mask_table_flat = mask_table.flatten()
    mask_wall_flat = mask_wall.flatten()
    y_full[mask_flat] = 1
    y_full[mask_couch_flat] = 1
    y_full[mask_table_flat] = 1
    y_full[mask_wall_flat] = 2
    # Now you have features for all pixels and labels for some based on your masks
    # Train the decision tree on the labeled data
    labeled_mask = y_full != 0 # Only train on pixels you've labeled
    X_train = features_all[labeled_mask]
    y_train = y_full[labeled_mask]

    print("Training labels:", np.unique(y_train, return_counts=True))

    tree = DecisionTreeClassifier(class_weight="balanced", max_depth=3, min_samples_split=10, random_state=42)
    tree.fit(X_train, y_train)
    print("Class labels:", tree.classes_)
    print("Class labels:", tree.classes_)

    print("Class distribution:", np.unique(y_train, return_counts=True))

    # Predict only on the masked pixels
    tree_predictions_masked = tree.predict(features_all[cluster_mask_flat2])

    # Create a full-size blank array and fill only the masked locations
    tree_predictions = np.full(image_rgb.shape[:2], fill_value=-1, dtype=int)

    # Fill in the predictions for the masked pixels
    tree_predictions_flat = tree_predictions.flatten()
    tree_predictions_flat[cluster_mask_flat2] = tree_predictions_masked

    # Reshape the predictions back to the original image shape
    tree_predictions = tree_predictions_flat.reshape(image_rgb.shape[:2])


    importances = tree.feature_importances_
    plt.barh(y=["softness", "softness-texture", "brightness"], width=importances)
    plt.title("Feature Importances from Decision Tree")
    plt.show()

    # Visualize the decision tree predictions
    plt.imshow(tree_predictions, cmap='tab10')
    plt.title("Decision Tree Predictions on Cluster")
    plt.axis("off")
    plt.show()

    unique_labels = np.unique(y_train)
    class_names = ["Cat-Bed",  "Wall"]
    # Adjust the indexing by subtracting 1 from unique_labels if class labels start from 1
    selected_class_names = [class_names[i-1 ] for i in unique_labels]

    # Visualize the tree
    plt.figure(figsize=(8, 6))
    plot_tree(tree, feature_names=["softness",  "softness-texture", "brightness"], class_names=selected_class_names, filled=True)
    plt.title("Refinement Tree for Cluster 1")
    plt.show()

    print(export_text(tree, feature_names=["softness", "softness-texture", "brightness"]))

    # Access the tree's structure
    tree_ = tree.tree_
    feature_names = [ "softness" , "softness-texture", "brightness"]

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
    mask_in_cluster2 = (labels_masked3 == best_cluster_label2)
    tree_preds_in_cluster = tree_predictions[mask_in_cluster2]
    unique_nodes = np.unique(tree_preds_in_cluster)

    for node in np.unique(tree_predictions):
        mask = (tree_predictions == node) & (labels_masked3 ==  best_cluster_label2)
        print(f"Node {node}: {np.count_nonzero(mask)} pixels in cluster")

    # Step 7: Dynamically create a color palette based on the number of unique nodes
    unique_nodes = np.unique(tree_predictions)
    num_nodes = len(unique_nodes)

    node_classes = tree.tree_.value  # This is a (n_nodes, n_classes) array
    class_names = selected_class_names  # Assuming binary classification: 0 for Other, 1 for Cat-Bed

    # Step 3: Print out the class for each node
    for node_id in range(node_classes.shape[0]):
        # Find the class with the maximum value in each node's value (distribution)
        class_index = np.argmax(node_classes[node_id])  # Index of the class with the highest count

        # Get the class name based on the index (assuming binary: 0 -> "Other", 1 -> "Cat-Bed")
        node_class = class_names[class_index-1]

        # Print the node ID and the most common class in that node
        print(f"Node {node_id}: Class = {node_class}")

    # Use the "tab20" colormap for up to 20 colors (can be adjusted for more nodes)
    color_map = cm.get_cmap("tab10", num_nodes)  # Use the tab20 colormap


    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#17becf', '#8c564b', '#e377c2', '#bcbd22', '#7f7f7f']

    # Create a ListedColormap for use with imshow
    from matplotlib.colors import ListedColormap
    #custom_cmap = ListedColormap(colors[:cv2.kmeans.n_clusters])

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
        node_mask = (tree_predictions == node_label) & (labels_masked3 ==  best_cluster_label2) #& (labels == target_cluster_id)

        # Apply color to those pixels
        overlay_image[node_mask] = rgb_255

        #Generate an informative legend
        class_index = np.argmax(node_classes[node_label])  # Index of the class with the highest count
        node_class = class_names[class_index-1]
        legend_handles.append(Patch(color=rgb_float, label=f'Node {node_label}: Class = {node_class}'))
        print("Size overlay: ")
        print(i)
        print(overlay_image.shape)

    plt.imshow(overlay_image)
    plt.title("Overlay Image Only")
    plt.axis("off")
    plt.show()

    # Plot the overlay on top of the original image
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.imshow(overlay_image, alpha=0.75)
    plt.title(f"Tree Nodes in Cluster { best_cluster_label2}")

    # Add the legend to the plot
    plt.legend(handles=legend_handles, loc='upper left', fontsize=12, title="Nodes")
    plt.axis("off")
    plt.show()
    return