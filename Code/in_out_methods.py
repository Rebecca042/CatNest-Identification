import cv2
import matplotlib.pyplot as plt

def read_image(path):
    image = cv2.imread(path)
    if image is None:
        print("Failed to load image.")
    else:
        print("Image loaded successfully.")
    return image

def read_image_as_rgb(path):
    image = read_image(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def show_image_rgb(image_rgb, title):
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()
    return

def show_image_comparsion(image_rgb, image_comparison, title1, title2):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_comparison)
    plt.title(title2)
    plt.axis("off")

    return

def show_image_flexible(images, titles):

    def check_input(number_images, number_titles):
        if number_images != number_titles:
            return False
        else:
            return True

    # Check to ensure the number of images and titles is the same

    if check_input(len(images), len(titles)):
        number_images = len(images)
        plt.figure(figsize=(12, 6))
        for index_image in range(len(images)):

            plt.subplot(1, number_images, index_image+1)
            plt.imshow(images[index_image])
            plt.title(titles[index_image])
            plt.axis("off")

    else:
        print("The number of titles ("+str(len(titles))+") does not correspond to the number of images ("+str(len(images))+").")

    return


def prepare_image_rgb(image_rgb, title):
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    return

def prepare_image_overlay(image_original, overlay, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_original)  # Original image
    ax.imshow(overlay, alpha=0.5, cmap='jet')  # Elliptical region overlay
    plt.title(title)
    plt.axis("off")
    return