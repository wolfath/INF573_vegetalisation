import matplotlib.pyplot as plt
import random   
import numpy as np
import rasterio
import cv2

from count.processing import NDVI, RGB, elevation, IR, extract_connected_components
from count.data import read_tif_image

def imshow(image, cmap=None, vmin=None, vmax=None, title=None):
    """Display an image whether its number of channels is 1, 3 or 5"""
    fig, ax = plt.subplots()
    ax.set_title(title)
    if cmap is None:
        cmap = 'viridis'
    if vmin is None:    
        vmin = np.min(image)
    if vmax is None:
        vmax = np.max(image)
    if image.shape[-1] == 5:
        ax.imshow(image[..., :3], cmap=cmap, vmin=vmin, vmax=vmax)

    else:
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.show()

def display_5channel_image(image):
    """Displays a 5 channel image assuming the 5 channels are RGB, NIR and Elevation"""
    rgb = RGB(image)
    ir = IR(image)
    elev = elevation(image)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('black')

    # Display the RGB image
    axs[0].imshow(rgb)
    axs[0].set_title("RGB Image", color='white')

    # Display the IR channel
    axs[1].imshow(ir, cmap='gray')
    axs[1].set_title("Infrared Channel", color='white')

    # Display the Elevation channel
    axs[2].imshow(elev, cmap='terrain')
    axs[2].set_title("Elevation Channel", color='white')

    plt.show()


def display_NDVI_RGB(image):
    """Displays the NDVI of an image next to IT"""
    ndvi = NDVI(image)
    rgb = RGB(image)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('black')
    ax1.imshow(rgb)
    ax1.set_title("RGB", color='white')
    ax2.imshow(ndvi, cmap='RdYlGn')
    ax2.set_title("NDVI", color='white')
    plt.show()

def display_samples(images,nb_samples: list) -> None:
    """
    Shows random images of the flair dataset
    """
    indices= random.sample(range(0, len(images)), nb_samples)
    fig, axs = plt.subplots(nrows = nb_samples, ncols = 6, figsize = (20, nb_samples * 6)); fig.subplots_adjust(wspace=0.0, hspace=0.01)
    fig.patch.set_facecolor('black')
    for u, idx in enumerate(indices):
        with rasterio.open(images[idx], 'r') as f:
            im = f.read([1,2,3]).swapaxes(0, 2).swapaxes(0, 1)
            imir = f.read([4])[0]
            imel = f.read([5])[0]
            imr= f.read([1])[0]
            img= f.read([2])[0]
            imb= f.read([3])[0]
        ax0 = axs[u][0] ; ax0.imshow(im);ax0.axis('off')
        ax1 = axs[u][1] ; ax1.imshow(imir);ax1.axis('off')
        ax2 = axs[u][2] ; ax2.imshow(imel);ax2.axis('off')
        ax3 = axs[u][3] ; ax3.imshow(imr);ax3.axis('off')
        ax4 = axs[u][4] ; ax4.imshow(img);ax4.axis('off')
        ax5 = axs[u][5] ; ax5.imshow(imb);ax5.axis('off')
        if u == 0:
            ax0.set_title('RVB Image', size=16,fontweight="bold",c='w')
            ax1.set_title('NIR Image', size=16,fontweight="bold",c='w')
            ax2.set_title('Elevation', size=16,fontweight="bold",c='w')
            ax3.set_title('Red', size=16,fontweight="bold",c='w')
            ax4.set_title('Green', size=16,fontweight="bold",c='w')
            ax5.set_title('Blue', size=16,fontweight="bold",c='w')
    for i in indices:
        print(images[i])

def display_colorized_connected_components(img):
    """
    Displays the connected components of a binary image in random colors.
    """
    img_binary = np.uint8(img)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_binary, connectivity=8)
    # Generate random colors for each component
    colors = [np.random.randint(0, 255, 3) for _ in range(nb_components)]
    # Initialize the resulting colorized image
    colorized_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # Assign a unique color for each component and fill the colorized image
    for i in range(1, nb_components):
        colorized_img[output == i] = colors[i - 1]
    imshow(cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB))
    print('Number of connected components: {}'.format(nb_components))

def display_colorized_connected_components_with_size_filtering(img, min_size=50):
    img_binary = np.uint8(img)
    connected_components_img,nb_components, output = extract_connected_components(img_binary, min_size=min_size)
    # Generate random colors for each component
    colors = [np.random.randint(0, 255, 3) for _ in range(nb_components)]
    # Initialize the resulting colorized image
    colorized_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # Assign a unique color for each component and fill the colorized image
    for i in range(1, nb_components):
        colorized_img[output == i] = colors[i - 1]
    imshow(cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB))
    print('Number of connected components: {}'.format(nb_components))

        
def display_result(image, coords_of_maximums, square_size=3):
    """
    Displays the result of the function that highlights the center of each tree
    """
    image2 = image.copy()
    for coord in coords_of_maximums:
        for i in range(coord[0] - square_size, coord[0] + square_size + 1):
            for j in range(coord[1] - square_size, coord[1] + square_size + 1):
                if i >= 0 and i < image.shape[0] and j >= 0 and j < image.shape[1]:
                    image2[i, j, :][:3] = [255, 0, 0]  # Set pixel color to red

    imshow(image2)



def test_method(method, images_link):
    for image_link in images_link:
        image = read_tif_image(image_link)
        coords_of_maximums = method(image)
        display_result(image, coords_of_maximums)




