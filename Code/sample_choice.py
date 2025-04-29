import numpy as np

def adjust_to_image(image_end, start, end):
    if start < 0:
        start = 0
    if end < 0:
        end = 0

    if start > image_end:
        start = image_end
    if end > image_end:
        end = image_end

    return start, end

def adjust_start_end(start, end):
    if start > end:
        end_tmp = end
        end = start
        start = end_tmp
    return start, end

def check_size(start_x, end_x, start_y, end_y, image_end_x, image_end_y):
    size_x = end_x - start_x
    size_y = end_y - start_y

    if size_x == 0:
        print("Area has size 0 in x-direction")
    if size_y == 0:
        print("Area has size 0 in y-direction")
    if size_x * size_y < 0.02 * image_end_x * image_end_y:
        print("Area has less than 5 % of original size")
    return

def debugging_final_size(start_x, end_x, start_y, end_y, image_end_x, image_end_y):
    print("The image shape is:")
    print("hight x width: " + str(image_end_y) + " x " + str(image_end_x))
    print("The shape of the area is:")
    print("X-direction: " + str(start_x) + " - " + str(end_x))
    print("Y-direction: " + str(start_y) + " - " + str(end_y))
    print("With cartesian coordinates:")
    start_x_cart, end_x_cart, start_y_cart, end_y_cart = 1100, 1400, image_end_y - end_y, image_end_y - start_y
    print("X-direction: " + str(start_x_cart) + " - " + str(end_x_cart))
    print("Y-direction: " + str(start_y_cart) + " - " + str(end_y_cart))
    return

def select_area(image_shape, start_x, end_x, start_y, end_y):
    # potentially: check for minimum size - TODO

    (image_end_y, image_end_x) = image_shape[:2]

    # Avoid errors - readjust the values in case they are outside
    start_x, end_x = adjust_to_image(image_end_x, start_x, end_x)
    start_y, end_y = adjust_to_image(image_end_y, start_y, end_y)

    # Improve readability - the selection goes from smaller -> larger values
    start_x, end_x = adjust_start_end(start_x, end_x)
    start_y, end_y = adjust_start_end(start_y, end_y)

    # Check if the size is ok
    check_size(start_x, end_x, start_y, end_y, image_end_x, image_end_y)

    # For debugging:
    debugging_final_size(start_x, end_x, start_y, end_y, image_end_x, image_end_y)

    return start_x, end_x, start_y, end_y

############################################
# Selection of the sample-shape
############################################

# Rectangle
def select_rectangle(sample_region, image_shape):
    # Example: Define your sample region as (y_min, y_max, x_min, x_max)
    y_min, y_max, x_min, x_max = sample_region
    height, width = image_shape[:2]
    rectangle_mask = np.zeros((height, width), dtype=bool)
    rectangle_mask[y_min:y_max, x_min:x_max] = True
    return rectangle_mask

# Ellipse
def select_ellipse(sample_region, image_shape):
    # Example: Define your sample region as (y_min, y_max, x_min, x_max)
    #sample_region = (650, 850,600, 990)

    # Extract the coordinates from the sample region
    y_min, y_max, x_min, x_max = sample_region

    # Calculate the center of the sample region
    center_y = (y_min + y_max) / 2
    center_x = (x_min + x_max) / 2

    # Semi-major and semi-minor axes: define them relative to the sample region
    semi_major_axis = (x_max - x_min) / 2  # Horizontal axis
    semi_minor_axis = (y_max - y_min) / 2  # Vertical axis

    # Create a mask for the elliptical region
    height, width = image_shape[:2]  # Replace with image size if different
    y, x = np.ogrid[:height, :width]

    # Apply the ellipse equation: ((x - center_x)^2 / semi_major^2) + ((y - center_y)^2 / semi_minor^2) <= 1
    ellipse_mask = ((x - center_x) ** 2 / semi_major_axis ** 2) + ((y - center_y) ** 2 / semi_minor_axis ** 2) <= 1

    return ellipse_mask

# TODO: Circle / polyhedron