import matplotlib.pyplot as plt

import in_out_methods as io
import sample_choice as sc
import clustering_w_sample as cl_sample
import select_features as sf
import clustering_decisiontree1 as cl_tree1
import clustering_decisiontree2 as cl_tree2
import clustering_decisiontree0 as cl_tree0

# Read in the image
image_path = "README.mdAI-Image-Cat1.png"
image_rgb = io.read_image_as_rgb(image_path)
image_rgb_shape = image_rgb.shape

# Show image to define a biased starting point (cat nest sample)
title = "Select the cat nest with a rectangle/ellipse"
io.show_image_rgb(image_rgb, title)

# Mark the biased sampling area on the original image
image_marked = image_rgb.copy()
image_marked_shape = image_rgb_shape

# potentially: reselect until happy :) - TODO
# TODO: reselect until happy :) - decide values flexibly via stdin

# Set the area - with some checks for safety
#start_x, end_x, start_y, end_y  = sc.select_area(image_marked_shape, 1100, 1400, 650, 800)
start_x, end_x, start_y, end_y  = sc.select_area(image_marked_shape, 1050, 1400, 650, 800)
sample_region = (start_y, end_y,start_x, end_x)


# Initialize Dictionary for output
type_dict = {-1: "None", 0: "Rectangle", 1: "Ellipse"}

# Set the mask for the rectangle and ellipse selection
rectangle_mask = sc.select_rectangle(sample_region, image_rgb_shape)
ellipse_mask = sc.select_ellipse(sample_region, image_rgb_shape)

# Display both results with the marked sampling area
title = "Biased Sample Area for K-Means Initialization - Rectangle"
io.prepare_image_overlay(image_rgb, rectangle_mask, title)

title = "Biased Sample Area for K-Means Initialization - Ellipse"
io.prepare_image_overlay(image_rgb, ellipse_mask, title)

plt.show()

########### K - means ######################
selected_k = 3
# Without bias
# Use the mask for a biased start
labels, segmented_image = cl_sample.cluster(selected_k, image_rgb)

title1 = "Original Image"
title2 = "Clustered Image (Cat Bed Biased)"
io.show_image_comparsion(image_rgb, segmented_image, title1, title2)

# Save the figure
plt.savefig("sample-" + type_dict[-1] + "-k-" + str(selected_k) + "-colorbased.png", bbox_inches='tight')

plt.tight_layout()
plt.show()

# Repetition - with a masked approach
# Initialize all necessary variables
mask_type = 1
mask = ellipse_mask

# Use the mask for a biased start
labels_masked1, segmented_image_masked1 = cl_sample.cluster_w_mask(mask, mask_type, selected_k, image_rgb)

title1 = "Original Image"
title2 = "Clustered Image (Cat Bed Biased)"
io.show_image_comparsion(image_rgb, segmented_image_masked1, title1, title2)

# Save the figure
plt.savefig("sample-" + type_dict[mask_type] + "-k-" + str(selected_k) + "-colorbased.png", bbox_inches='tight')

plt.tight_layout()
#plt.show()

########### Refine the Clustering #################
### Use of brightness and softness ################
print("Calculate brightness and softness to refine the clustering.")
# Calculate Brightness and Softness
brightness, softness = sf.select_features(image_rgb)

soft_hard_map, bright_dark_map = sf.define_masks(brightness, softness, image_rgb)

# Display the Brightness and Softness along with the Original Image
titles = ["Original Image", "Hard vs Soft (White = Hard)", "Bright vs Dark (White = Bright)"]
images = [image_rgb, soft_hard_map, bright_dark_map]
io.show_image_flexible(images, titles)

# Save the figure
plt.savefig("clustering_comparison_Features.png", bbox_inches='tight')

plt.tight_layout()
#plt.show()


### Start of the Clustering #########################
# Reshape to use as feature
brightness = brightness.reshape(-1,1)
softness = softness.reshape(-1,1)

labels_features1, segmented_image_features1 = cl_sample.cluster_w_features([softness, brightness], selected_k, image_rgb)
# --- Visualize ---
title1 = "Clustered Image (Cat Bed Biased)"
title2 = "Clustering with Features: Softness, Brightness"
io.show_image_comparsion(segmented_image_masked1, segmented_image_features1, title1, title2)
plt.savefig("sample-" + type_dict[mask_type] + "-k-" + str(selected_k) + "-wFeatures.png", bbox_inches='tight')
#plt.show()

#### Combine Features and the Biased Sampling ####

# Use the features and mask from the tasks before
labels_features_masked1, segmented_image_features_masked1 = cl_sample.cluster_w_features_mask([softness, brightness], mask, mask_type, selected_k, image_rgb)

# Display the Brightness and Softness along with the Original Image
titles = [ "Clustering with Features: Softness, Brightness", "Combined Approach"]
images = [ segmented_image_features1, segmented_image_features_masked1]
io.show_image_flexible(images, titles)
plt.savefig("sample-" + type_dict[mask_type] + "-k-" + str(selected_k) + "-CombinedApproach.png", bbox_inches='tight')
#plt.show()

#### Add another feature ##########################
color_distance_scaled = sf.select_color_distance(mask, image_rgb)
labels_features_masked2, segmented_image_features_masked2 = cl_sample.cluster_w_features_mask([softness, brightness, color_distance_scaled], mask, mask_type, selected_k, image_rgb)
# Display the Brightness and Softness along with the Original Image
titles = ["Combined Approach (Softness, Brightness)",  "Combined Approach (Softness, Brightness, Color Distance)"]
images = [segmented_image_features_masked1, segmented_image_features_masked2]
io.show_image_flexible(images, titles)
plt.savefig("sample-" + type_dict[mask_type] + "-k-" + str(selected_k) + "-CombinedApproach-3Features.png", bbox_inches='tight')
#plt.show()


print("Testing different window sizes for the features.")
### Options for the Features - Brightness, Softness ###
### Window-Size #######################################

# First Option: 3 Clusters + 3 Window sizes (2, 5, 10)
window_sizes = [2, 5, 10]
selected_k = 3
labels_collect1 = []
segmented_image_collect1 = []
soft_hard_map_collect1 = []
titels_collect1 = []
for window_size in window_sizes:
    # Update brightness and softness to the specified window size
    brightness, softness = sf.select_features_windowed(image_rgb, window_size)
    soft_hard_map, bright_dark_map = sf.define_masks(brightness, softness, image_rgb)

    # Reshape to use as feature
    brightness = brightness.reshape(-1,1)
    softness = softness.reshape(-1,1)

    labels_tmp, segmented_image_tmp = cl_sample.cluster_w_features_mask([softness, brightness], mask, mask_type, selected_k, image_rgb)

    labels_collect1.append(labels_tmp)
    segmented_image_collect1.append(segmented_image_tmp)
    soft_hard_map_collect1.append(soft_hard_map)
    titels_collect1.append("Window-size: "+str(window_size)+"")

titles = titels_collect1
images = soft_hard_map_collect1
io.show_image_flexible(images, titles)
plt.suptitle("Effect of Window Sizes on the Softness", fontsize=16)
plt.savefig("softness-MultipleWindowSizes.png", bbox_inches='tight')
#plt.show()

titles = titels_collect1
images = segmented_image_collect1
io.show_image_flexible(images, titles)
plt.suptitle("Combined Approach (Softness, Brightness with flexible window-size)", fontsize=16)
plt.savefig("sample-" + type_dict[mask_type] + "-k-" + str(selected_k) + "-MultipleWindowSizes.png", bbox_inches='tight')
#plt.show()

# Second Option: 2 Clusters + 3 Window sizes (2, 5, 10)
window_sizes = [2, 5, 10]
selected_k = 2
labels_collect1 = []
segmented_image_collect1 = []
soft_hard_map_collect1 = []
titels_collect1 = []
for window_size in window_sizes:
    # Update brightness and softness to the specified window size
    brightness, softness = sf.select_features_windowed(image_rgb, window_size)
    soft_hard_map, bright_dark_map = sf.define_masks(brightness, softness, image_rgb)

    # Reshape to use as feature
    brightness = brightness.reshape(-1,1)
    softness = softness.reshape(-1,1)

    labels_tmp, segmented_image_tmp = cl_sample.cluster_w_features_mask([softness, brightness], mask, mask_type, selected_k, image_rgb)

    labels_collect1.append(labels_tmp)
    segmented_image_collect1.append(segmented_image_tmp)
    soft_hard_map_collect1.append(soft_hard_map)
    titels_collect1.append("Window-size: "+str(window_size)+"")

titles = titels_collect1
images = segmented_image_collect1
io.show_image_flexible(images, titles)
plt.suptitle("Combined Approach (Softness, Brightness with flexible window-size)", fontsize=16)
plt.savefig("sample-" + type_dict[mask_type] + "-k-" + str(selected_k) + "-MultipleWindowSizes.png", bbox_inches='tight')
#plt.show()


# Third Option: 5 Clusters + 3 Window sizes (2, 5, 10)
window_sizes = [2, 5, 10]
selected_k = 5
labels_collect1 = []
segmented_image_collect1 = []
soft_hard_map_collect1 = []
titels_collect1 = []
for window_size in window_sizes:
    # Update brightness and softness to the specified window size
    brightness, softness = sf.select_features_windowed(image_rgb, window_size)
    soft_hard_map, bright_dark_map = sf.define_masks(brightness, softness, image_rgb)

    # Reshape to use as feature
    brightness = brightness.reshape(-1,1)
    softness = softness.reshape(-1,1)

    labels_tmp, segmented_image_tmp = cl_sample.cluster_w_features_mask([softness, brightness], mask, mask_type, selected_k, image_rgb)

    labels_collect1.append(labels_tmp)
    segmented_image_collect1.append(segmented_image_tmp)
    soft_hard_map_collect1.append(soft_hard_map)
    titels_collect1.append("Window-size: "+str(window_size)+"")

titles = titels_collect1
images = segmented_image_collect1
io.show_image_flexible(images, titles)
plt.suptitle("Combined Approach (Softness, Brightness with flexible window-size)", fontsize=16)
plt.savefig("sample-" + type_dict[mask_type] + "-k-" + str(selected_k) + "-MultipleWindowSizes.png", bbox_inches='tight')
#plt.show()

########### Refine with the use of a binary-tree #########
# Use only the color for the clustering
# Refine the relevant cluster with a binary-tree
selected_k = 2
max_iter = 100  # Reduce the number of max_iterations to avoid overly consistent clusters

cl_tree0.clustering_and_tree(ellipse_mask, mask_type, selected_k, max_iter, image_rgb)

########### Refine with the use of a binary-tree #########
# Use only the color for the clustering
# Refine the relevant cluster with a binary-tree
selected_k = 2
max_iter = 100 # Reduce the number of max_iterations to avoid overly consistent clusters


cl_tree1.clustering_and_tree(ellipse_mask, mask_type, selected_k, max_iter, image_rgb)


### Another comparison - with a new Feature #######################
# Clustering - use of the Texture feature #
selected_k = 5
window_size = 5

# Update brightness and softness to the specified window size
softness_std = sf.select_softness_std(image_rgb, window_size)
brightness, softness = sf.select_features_windowed(image_rgb, window_size)
soft_hard_map, bright_dark_map = sf.define_masks(brightness, softness, image_rgb)

# Reshape to use as feature
brightness = brightness.reshape(-1,1)
softness = softness.reshape(-1,1)

labels_new_feature, segmented_image_new_feature = cl_sample.cluster_w_features_mask([softness, brightness, softness_std], mask, mask_type, selected_k, image_rgb)

# --- Visualize ---
title1 = "Clustering with Features: Softness, Brightness"
title2 = "Clustering with Features: Softness, Brightness, Texture"
io.show_image_comparsion( segmented_image_features1, segmented_image_new_feature, title1, title2)
plt.savefig("sample-" + type_dict[mask_type] + "-k-" + str(selected_k) + "-wFeaturesNew.png", bbox_inches='tight')
plt.show()

# Decision Tree - Use of the Texture Feature #
cl_tree2.clustering_and_tree2(ellipse_mask, mask_type, selected_k, max_iter, image_rgb)


### Another comparison - with a new Feature #######################
# Clustering - use of the Texture feature #
ks = [2, 3, 5]
segemented_image_new_feature_multiple_k = []
titles_multiple_k = []
for selected_k in ks:
    window_sizes = 5

    # Update brightness and softness to the specified window size
    softness_std = sf.select_softness_std(image_rgb, window_size)
    brightness, softness = sf.select_features_windowed(image_rgb, window_size)
    soft_hard_map, bright_dark_map = sf.define_masks(brightness, softness, image_rgb)

    # Reshape to use as feature
    brightness = brightness.reshape(-1,1)
    softness = softness.reshape(-1,1)

    labels_new_feature_tmp, segmented_image_new_feature_tmp = cl_sample.cluster_w_features_mask([softness, brightness, softness_std], mask, mask_type, selected_k, image_rgb)

    segemented_image_new_feature_multiple_k.append(segmented_image_new_feature_tmp)
    titles_multiple_k.append("k = " +str(selected_k)+"")

# --- Visualize ---
io.show_image_flexible(segemented_image_new_feature_multiple_k, titles_multiple_k)
plt.suptitle("Clustering with Features: Softness, Brightness, Texture (flexible k)", fontsize=16)
plt.savefig("sample-" + type_dict[mask_type] + "-k-" + str(selected_k) + "-NewFeature-multipleK.png", bbox_inches='tight')
plt.show()


