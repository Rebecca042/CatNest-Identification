import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import in_out_methods as io

def select_starting_sample(mask, type, image_rgb):
    # Set the sample region with the chosen mask
    if type == 0:
        print("The decision is for the ellipse sample.")
        # sample_region = image_rgb[start_y:end_y, start_x:end_x]
        sample_region = image_rgb[mask]
    if type == 1:
        print("The decision is for the ellipse sample.")
        sample_region = image_rgb[mask]
    sample_pixels = sample_region.reshape(-1, 3)
    return sample_pixels

def cluster_w_mask(mask, mask_type, selected_k, image_rgb):
    print("Clustering with an initialization based on the selected sample is executed.")
    #Apply mask to image - select sample pixels
    sample_pixels = select_starting_sample(mask, mask_type, image_rgb)

    # Reshape the whole image for clustering
    image_pixels = image_rgb.reshape(-1, 3)

    # Perform k-means with biased initialization using the sample region
    k = selected_k
    kmeans = KMeans(n_clusters=k, init=sample_pixels[np.random.choice(len(sample_pixels), k, replace=False)], n_init=1, max_iter=300, random_state=42)
    kmeans.fit(image_pixels)

    # Reshape labels to match image
    labels = kmeans.labels_.reshape(image_rgb.shape[:2])

    # Visualize clustered result
    segmented_image = kmeans.cluster_centers_[labels]
    segmented_image = segmented_image.astype(np.uint8)

    return labels, segmented_image

def cluster_w_mask2(mask, mask_type, selected_k, max_iter, image_rgb):
    print("Clustering with an initialization based on the selected sample is executed.")
    #Apply mask to image - select sample pixels
    sample_pixels = select_starting_sample(mask, mask_type, image_rgb)

    # Reshape the whole image for clustering
    image_pixels = image_rgb.reshape(-1, 3)

    # Perform k-means with biased initialization using the sample region
    k = selected_k
    kmeans = KMeans(n_clusters=k, init=sample_pixels[np.random.choice(len(sample_pixels), k, replace=False)], n_init=1, max_iter=max_iter, random_state=42)
    kmeans.fit(image_pixels)

    # Reshape labels to match image
    labels = kmeans.labels_.reshape(image_rgb.shape[:2])

    # Visualize clustered result
    segmented_image = kmeans.cluster_centers_[labels]
    segmented_image = segmented_image.astype(np.uint8)

    return labels, segmented_image

def cluster( selected_k, image_rgb):
    print("Basic clustering is executed.")
    # Reshape the whole image for clustering
    image_pixels = image_rgb.reshape(-1, 3)

    # Perform k-means with biased initialization using the sample region
    k = selected_k
    kmeans = KMeans(n_clusters=k, n_init=1, max_iter=300, random_state=42)
    kmeans.fit(image_pixels)

    # Reshape labels to match image
    labels = kmeans.labels_.reshape(image_rgb.shape[:2])

    # Visualize clustered result
    segmented_image = kmeans.cluster_centers_[labels]
    segmented_image = segmented_image.astype(np.uint8)

    return labels, segmented_image

def cluster_w_features(features, selected_k, image_rgb):
    def repeat_feature_3d(feature_1d):
        #Generate 3D features from 1D features
        # Example: [0.5, 0.8] → [[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]]

        feature_1d = np.asarray(feature_1d).reshape(-1, 1)
        #feature_3d = np.repeat(feature_1d, repeats=3, axis=1)
        return feature_1d

    print("Clustering with features is executed.")
    # --- Combine Features ---
    image_pixels = image_rgb.reshape(-1, 3)
    all_features = [image_pixels]
    for feature in features:
        # Sanity checks
        print("Feature shape:", feature.shape)
        print("Unique rows:", np.unique(feature, axis=0).shape)

        # Make it 3D
        feature3d = repeat_feature_3d(feature)
        all_features.append(feature3d)

    k = selected_k
    X = np.hstack(all_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Run Clustering ---
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)

    #labels = kmeans.fit_predict(X)
    kmeans.fit(X_scaled)
    # Reshape labels to match image
    labels = kmeans.labels_.reshape(image_rgb.shape[:2])
    cluster_centers_scaled = kmeans.cluster_centers_

    # Inverse transform to original scale
    cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)
    print(cluster_centers_original)
    # Visualize clustered result
    # Extract only RGB part of the cluster centers (first 3 values)
    cluster_colors = np.clip(cluster_centers_original[:, :3], 0, 255).astype(np.uint8)

    # Recolor image with cluster colors
    segmented_image = cluster_colors[kmeans.labels_].reshape(image_rgb.shape)

    return labels, segmented_image


def cluster_w_features_mask(features, mask, mask_type, selected_k, image_rgb):
    def repeat_feature_3d(feature_1d):
        #Generate 3D features from 1D features
        # Example: [0.5, 0.8] → [[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]]

        feature_1d = np.asarray(feature_1d).reshape(-1, 1)
        feature_3d = np.repeat(feature_1d, repeats=3, axis=1)
        return feature_1d

    print("Clustering with the combined approach - features and mask - is executed.")
    # --- Combine Features ---
    image_pixels = image_rgb.reshape(-1, 3)
    all_features = [image_pixels]
    for feature in features:
        # Sanity checks
        print("Feature shape:", feature.shape)
        print("Unique rows:", np.unique(feature, axis=0).shape)
        # Make it 3D
        feature3d = repeat_feature_3d(feature)
        all_features.append(feature3d)

    k = selected_k
    X = np.hstack(all_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use the mask to select features
    sampled_features = X[mask.astype(bool).flatten()]

    # --- Run Clustering ---
    kmeans = KMeans(n_clusters=k, init=sampled_features[np.random.choice(len(sampled_features), k, replace=False)], n_init=10, max_iter=300, random_state=42)

    #labels = kmeans.fit_predict(X)
    kmeans.fit(X_scaled)
    # Reshape labels to match image
    labels = kmeans.labels_.reshape(image_rgb.shape[:2])
    cluster_centers_scaled = kmeans.cluster_centers_

    # Inverse transform to original scale
    cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)
    print(cluster_centers_original)
    # Visualize clustered result
    # Extract only RGB part of the cluster centers (first 3 values)
    cluster_colors = np.clip(cluster_centers_original[:, :3], 0, 255).astype(np.uint8)

    # Recolor image with cluster colors
    segmented_image = cluster_colors[kmeans.labels_].reshape(image_rgb.shape)

    return labels, segmented_image



