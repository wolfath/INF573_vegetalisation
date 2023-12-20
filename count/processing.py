import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import maximum_filter
from scipy import stats
from scipy.ndimage.morphology import binary_dilation
from PIL import Image
from CNN.utils_cnn import transform_to_only_trees_mask
import matplotlib.patches as mpatches
from skimage.measure import find_contours
import cv2
from skimage import io, color, filters, draw, measure




def elevation(image):
    """
    Returs the elevation channel of a five channel image.
    """
    return image[:,:,4]

def IR(image):
    """
    Returns the IR channel of a five channel image.
    """
    return image[:,:,3]


def RGB(image):
    """
    Returns the RGB of a five channel image.
    """
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    RGB = np.stack((R, G, B), axis=-1)
    return RGB

def NDVI(image):
    """
    Returns the NDVI of a five channel image.
    """
    R = image[:,:,0].astype(float)
    IR = image[:,:,3].astype(float)
    NDVI = (IR - R) / (IR + R + 1e-6)
    return NDVI


def binary_threshold(image, threshold_min=0, threshold_max=1e7):
    """
    Returns the image with the pixels below the threshold set to 0.
    """
    binary = np.zeros_like(image)
    binary[image > threshold_min] = 1
    binary[image > threshold_max] = 0
    return binary

def binary_map(image, minNDVI=0, maxElev=70, minElev=10):
    """
    Applies the binary threshold about both the elevation and the NDVI on a given 5 channel image.
    """
    ndvi = NDVI(image)
    elev = elevation(image)
    binary = np.zeros_like(ndvi)
    binary[ndvi > minNDVI] = 1
    binary[elev > maxElev] = 0
    binary[elev < minElev] = 0
    return binary

def indice(liste, element):
    for i in range(len(liste)):
        if liste[i] == element:
            return i
    return -1

def new_output(true_outputs, output):
    new_outputs =  []
    for i in range(1,len(true_outputs)+1):
        new_outputs.append(i)
    new_output = new_outputs[indice(true_outputs,output)]
    

def extract_connected_components(image, min_size=50):
    """
    Extracts the connected components of a binary image.
    image : binary image
    min_size : minimum size of the connected components to keep
    """
    binary = np.uint8(image.copy())
    # Find connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # Extract sizes and update the number of components
    sizes = stats[1:, cv2.CC_STAT_AREA]
    nb_components -= 1
    # Initialize the answer image
    connected_components_img = np.zeros_like(binary)
    # Keep components above the minimum size
    true_component=[]
    true_outputs=[]
    real_nb_components = 0
    for i in range(1, nb_components+1):
        if sizes[i-1] >= min_size:
            connected_components_img[output == i] = 255
            real_nb_components += 1
            true_component.append(True)
        else:  
            true_component.append(False)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if not(true_component[output[i,j]-1]):
                output[i,j]=0
            else : 
                if output[i,j] not in true_outputs:
                    true_outputs.append(output[i,j])

    possible_new_outputs =  []
    for i in range(1,len(true_outputs)+1):
        possible_new_outputs.append(i)

    new_outputs = np.zeros_like(output)
    for i in range(new_outputs.shape[0]):
        for j in range(new_outputs.shape[1]):
            if output[i,j]!=0:
                new_outputs[i,j] = possible_new_outputs[indice(true_outputs,output[i,j])]
    
    return connected_components_img, real_nb_components, new_outputs



def coords_max_ndvi_component(image, min_size=50):
    """
    Returns the image with the maximum NDVI value in each connected component colored.
    image : 5 channel image
    square_size : size of the square around the maximum value
    min_size : minimum size of the connected components to keep
    """
    # Initialiser l'image résultante en noir et blanc (img_result)
    ndvi = NDVI(image)
    img_filtered, nb_components, output = extract_connected_components(binary_map(image), min_size=min_size)
    
    coords = []
    # Parcourir chaque composant connecté
    for i in range(1, nb_components+1):
        # Initialiser une image pour le composant connecté courant
        img_result2 = np.zeros_like(ndvi)
        
        # Créer un masque pour le composant connecté
        component_mask = (output == i)
        
        # Remplir l'image img_result2 avec les valeurs NDVI du composant connecté
        img_result2[component_mask] = ndvi[component_mask]

        # Trouver les coordonnées (i, j) du pixel avec la valeur maximale dans le composant connecté
        max_ndvi_coord = np.unravel_index(np.argmax(img_result2), img_result2.shape)

        coords.append(max_ndvi_coord)

    return coords



image = np.zeros((100, 100))

def local_maximums(image, size=30):
    # Apply maximum filter to find local maxima
    local_maxima = maximum_filter(image, size) == image
    coords_local_maximum=[]
    local_maxima[image==0]=False
    for i in range(local_maxima.shape[0]):
        for j in range(local_maxima.shape[1]):
            if local_maxima[i,j]:
                coords_local_maximum.append((i,j))
    return coords_local_maximum

def local_maximums_of_ndvi_connexe_components(image,min_size=50, local_max_size=30):
    ndvi = NDVI(image)
    img_filtered, nb_components, output = extract_connected_components(binary_map(image), min_size=min_size)
    coords = []
    # Parcourir chaque composant connecté
    for i in range(1, nb_components+1):
        # Initialiser une image pour le composant connecté courant
        img_result2 = np.zeros_like(ndvi)
        # Créer un masque pour le composant connecté
        component_mask = (output == i)
        # Remplir l'image img_result2 avec les valeurs NDVI du composant connecté
        img_result2[component_mask] = ndvi[component_mask]
        coords += local_maximums(img_result2, size=local_max_size)
    return coords

def binary_masks_from_mask(msk, smoothing=15):
    binary_masks = []
    for i in range(1, 7):
        mask = msk == i
        dilated_mask = binary_dilation(mask, iterations=smoothing)  # Increase the iterations value for stronger dilation
        binary_masks.append(dilated_mask)
    return binary_masks

def binary_masks_from_mask_train(msk, smoothing=15):
    binary_masks = []
    for i in range(0, 7):
        mask = msk == i
        dilated_mask = binary_dilation(mask, iterations=smoothing)  # Increase the iterations value for stronger dilation
        binary_masks.append(dilated_mask)
    return binary_masks








#------------------------------------------------------------      
def local_maximums_of_ndvi_connexe_components(image,min_size=50, local_max_size=30, original_image=image):
    print("start")
    ndvi = NDVI(image)
    print("ndvi done")
    img_filtered, nb_components, output = extract_connected_components(binary_map(image), min_size=min_size)
    coords = []
    # Parcourir chaque composant connecté
    for i in range(1, nb_components):
        print(i)
        # Initialiser une image pour le composant connecté courant
        img_result2 = np.zeros_like(ndvi)
        # Créer un masque pour le composant connecté
        component_mask = (output == i)
        # Remplir l'image img_result2 avec les valeurs NDVI du composant connecté
        img_result2[component_mask] = ndvi[component_mask]
        coords += local_maximums(img_result2, size=local_max_size, original_image = original_image)
    return coords






def local_maximums_clean(image, size=30, original_image=image):
    kernel = np.ones((5, 5), np.uint8)
    # image = np.flipud(image)
    #image =  binary_dilation(image, structure=kernel)
    maximum_filter_image = maximum_filter(image, size)
    # Apply maximum filter to find local maxima
    local_maxima = maximum_filter(image, size) == image
    coords_local_maximum = []
    local_maxima[image == 0] = False

    # Extract coordinates of local maxima
    for i in range(local_maxima.shape[0]):
        for j in range(local_maxima.shape[1]):
            if local_maxima[i, j]:
                coords_local_maximum.append((i, j))

    num_pixels_local_maxima = np.sum(local_maxima)
    

    kernel = np.ones((7, 7), np.uint8)

    trees_locations = binary_dilation(local_maxima, structure=kernel)


    # Créer une copie de l'image originale pour superposer les points
    image_with_points = original_image.copy()

    # Superposer les points en rouge
    square_size = 3
    for coord in coords_local_maximum:
        for i in range(coord[0] - square_size, coord[0] + square_size + 1):
            for j in range(coord[1] - square_size, coord[1] + square_size + 1):
                if i >= 0 and i < image.shape[0] and j >= 0 and j < image.shape[1]:
                    image_with_points[i, j] = [255, 0, 0]  # Set pixel color to red
                    True
    

   

    # Récupérer les coordonnées x et y des points d'intérêt
    x_coords = [coord[1] for coord in coords_local_maximum]
    y_coords = [coord[0] for coord in coords_local_maximum]


    return coords_local_maximum



#------------------------------------------------------------
def extract_connected_components_clean(image, minimum_CC_size=50):
    """
    Extracts the connected components of a binary image.
    image : binary image
    min_size : minimum size of the connected components to keep
    """
    binary = np.uint8(image.copy())
    # Find connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # Extract sizes and update the number of components
    sizes = stats[1:, -1]
    nb_components -= 1
    # Initialize the answer image
    connected_components_img = np.zeros_like(binary)
    # Keep components above the minimum size
    true_component=[]
    true_outputs=[]
    real_nb_components = 0
    for i in range(nb_components):
        if sizes[i] >= minimum_CC_size:
            connected_components_img[output == i + 1] = 255
            real_nb_components += 1
            true_component.append(True)
        else:  
            true_component.append(False)
    if nb_components>0:      
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if not(true_component[output[i,j]-1]):
                    output[i,j]=0
                else : 
                    if output[i,j] not in true_outputs:
                        true_outputs.append(output[i,j])
    
    possible_new_outputs =  []
    for i in range(1,len(true_outputs)+1):
        possible_new_outputs.append(i)
    
    new_outputs = np.zeros_like(output)
    if len(possible_new_outputs) > 0:
        # print("new outputs shape : ", new_outputs.shape)
        for i in range(new_outputs.shape[0]):
            for j in range(new_outputs.shape[1]):
                # print("i : ", i, " j : ", j)
                # print("indice : ", indice(true_outputs,output[i,j]))
                # print("len possible new outputs : ", len(possible_new_outputs))
                new_outputs[i,j] = possible_new_outputs[indice(true_outputs,output[i,j])]
            
    
    return connected_components_img, real_nb_components, new_outputs





def binary_map_clean(image, ndvi_treshold, elevation_threshold_min, elevation_threshold_max):
    """
    Applies the binary threshold about both the elevation and the NDVI on a given 5 channel image.
    """
    ndvi = NDVI(image)
    elev = elevation(image)
    binary = np.zeros_like(ndvi)
    binary[ndvi > ndvi_treshold] = 1
    
    binary[elev > elevation_threshold_max] = 0
    binary[elev < elevation_threshold_min] = 0

    return binary







def local_maximums_clean(image, maximum_filter_size=30):
    kernel = np.ones((5, 5), np.uint8)
    # image = np.flipud(image)
    #image =  binary_dilation(image, structure=kernel)
    # maximum_filter_image = maximum_filter(image, maximum_filter_size)
    # Apply maximum filter to find local maxima
    local_maxima = maximum_filter(image, maximum_filter_size) == image
    coords_local_maximum = []
    local_maxima[image == 0] = False

    # Extract coordinates of local maxima
    for i in range(local_maxima.shape[0]):
        for j in range(local_maxima.shape[1]):
            if local_maxima[i, j]:
                coords_local_maximum.append((i, j))  

    return coords_local_maximum






def local_maximums_of_ndvi_connexe_components_clean(image, ndvi_treshold, elevation_threshold_min, elevation_threshold_max, maximum_filter_size, minimum_CC_size):
    ndvi = NDVI(image)
    # img_filtered, nb_components, output = extract_connected_components_clean(binary_map_clean(image, ndvi_treshold, elevation_threshold_min, elevation_threshold_max), minimum_CC_size=minimum_CC_size)
    img_filtered, nb_components, output = extract_connected_components_clean(binary_map_clean(image, ndvi_treshold, elevation_threshold_min, elevation_threshold_max), minimum_CC_size= minimum_CC_size)
    coords = []
    # Parcourir chaque composant connecté
    for i in range(1, nb_components):
        # Initialiser une image pour le composant connecté courant
        img_result2 = np.zeros_like(ndvi)
        # Créer un masque pour le composant connecté
        component_mask = (output == i)
        # Remplir l'image img_result2 avec les valeurs NDVI du composant connecté
        img_result2[component_mask] = ndvi[component_mask]
        coords += local_maximums_clean(img_result2, maximum_filter_size=maximum_filter_size)
    return coords

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


def show_trees(tif_image, ndvi_treshold=0, elevation_threshold_min=0, elevation_threshold_max=80, maximum_filter_size=15, minimum_CC_size=150, square_size=3, output_name="output_image"):
    coords_of_maximums=local_maximums_of_ndvi_connexe_components_clean(tif_image, ndvi_treshold = ndvi_treshold, elevation_threshold_min = elevation_threshold_min, elevation_threshold_max=elevation_threshold_max, maximum_filter_size=maximum_filter_size, minimum_CC_size=minimum_CC_size)
    image_rgb = RGB(tif_image)
    image_with_trees = tif_image.copy()
    for coord in coords_of_maximums:
        for i in range(coord[0] - square_size, coord[0] + square_size + 1):
            for j in range(coord[1] - square_size, coord[1] + square_size + 1):
                if i >= 0 and i < image_rgb.shape[0] and j >= 0 and j < image_rgb.shape[1]:
                    image_with_trees[i, j, :][:3] = [255, 0, 0]  # Set pixel color to red
    
    imshow(image_with_trees)
    

    output_path = "output/" + output_name + ".jpg"
    # cv2.imwrite(output_path, (RGB(image_with_trees)).astype(np.uint8))


    image_to_save = Image.fromarray(RGB(image_with_trees))

    image_to_save.save(output_path)
    print("Image saved in " + output_path)

    return (image_with_trees, coords_of_maximums)



def show_trees2(tif_image, ndvi_treshold=0, elevation_threshold_min=0, elevation_threshold_max=80, maximum_filter_size=15, minimum_CC_size=150, square_size=3, output_name="output_image"):
    coords_of_maximums=local_maximums_of_ndvi_connexe_components_clean(tif_image, ndvi_treshold = ndvi_treshold, elevation_threshold_min = elevation_threshold_min, elevation_threshold_max=elevation_threshold_max, maximum_filter_size=maximum_filter_size, minimum_CC_size=minimum_CC_size)
    image_rgb = RGB(tif_image)
    image_with_trees = tif_image.copy()
    for coord in coords_of_maximums:
        for i in range(coord[0] - square_size, coord[0] + square_size + 1):
            for j in range(coord[1] - square_size, coord[1] + square_size + 1):
                if i >= 0 and i < image_rgb.shape[0] and j >= 0 and j < image_rgb.shape[1]:
                    image_with_trees[i, j, :][:3] = [255, 0, 0]  # Set pixel color to red

    return (image_with_trees, coords_of_maximums)




def isFarEnough(coords1, coords2, threshold):
        return np.linalg.norm(np.array(coords1) - np.array(coords2)) > threshold


def show_trees_from_mask(tif_image, old_mask, ndvi_treshold, elevation_threshold_min, elevation_threshold_max, maximum_filter_size, minimum_CC_size, output_name):
    new_mask = transform_to_only_trees_mask(old_mask)
    binary_masks = binary_masks_from_mask(new_mask, smoothing=5)

    tab_coords_of_maximums = []
    final_image_with_trees = np.copy(tif_image)
    for numero_of_the_class, binary_mask in enumerate(binary_masks):
        current_image = np.zeros_like(tif_image)  
        current_image[binary_mask] = tif_image[binary_mask]

        current_image_with_trees, current_coords_of_maximums = show_trees2(current_image, ndvi_treshold=ndvi_treshold[numero_of_the_class] , elevation_threshold_min=elevation_threshold_min[numero_of_the_class], elevation_threshold_max=elevation_threshold_max[numero_of_the_class], maximum_filter_size=maximum_filter_size[numero_of_the_class], minimum_CC_size=minimum_CC_size[numero_of_the_class], output_name = output_name)

        tab_coords_of_maximums.append(current_coords_of_maximums)


    valid_coords_of_maximums = []
    tab_numero_of_the_class = []
    for numero_of_the_class, coords_of_maximums in enumerate(tab_coords_of_maximums):
        for coord in coords_of_maximums:
            if(len(valid_coords_of_maximums)==0):
                valid_coords_of_maximums.append(coord)
                tab_numero_of_the_class.append(numero_of_the_class)

            else:
                for valid_coord in valid_coords_of_maximums:
                    booleanFarEnough = True
                    if(not isFarEnough(valid_coord, coord, 10)):
                        booleanFarEnough = False
                        break
                if(booleanFarEnough):
                    valid_coords_of_maximums.append(coord)
                    tab_numero_of_the_class.append(numero_of_the_class)

    colors = [
        [0, 255, 0],
        [255, 0, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [128, 128, 128]
    ]


    square_size=3       
    final_image_with_trees = np.copy(tif_image)  
    for numero_of_coord, coord in enumerate(valid_coords_of_maximums):
            for i in range(coord[0] - square_size, coord[0] + square_size + 1):
                for j in range(coord[1] - square_size, coord[1] + square_size + 1):
                    if i >= 0 and i < tif_image.shape[0] and j >= 0 and j < tif_image.shape[1]:
                        final_image_with_trees[i, j, :][:3] = colors[tab_numero_of_the_class[numero_of_coord]]  # Set pixel color to red




    #display the image with trees + legend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [4, 1]})


    ax1.imshow(final_image_with_trees[..., :3], cmap='viridis') 
    ax1.axis('off')


    class_names = ['coniferous', 'deciduous', 'brushwood', 'vineyard', 'herbaceous vegetation', 'ligneous', 'other']

    for color, label in zip(colors, class_names):
        color_normalized = np.array(color) / 255.0
        ax2.scatter([], [], color=color_normalized, label=label, s=100)

    ax2.axis('off')

    ax2.legend(loc='center', bbox_to_anchor=(0, 0.5))
    
    plt.show()



    output_path = "output/" + output_name + ".jpg"


    image_to_save = Image.fromarray(RGB(final_image_with_trees))

    image_to_save.save(output_path)





# def show_trees_from_mask2(tif_image, old_mask, ndvi_treshold, elevation_threshold_min, elevation_threshold_max, maximum_filter_size, minimum_CC_size, output_name):
#     new_mask = transform_to_only_trees_mask(old_mask)
#     binary_masks = binary_masks_from_mask(new_mask, smoothing=1)

#     tab_coords_of_maximums = []
#     final_image_with_trees = np.copy(tif_image)
#     for numero_of_the_class, binary_mask in enumerate(binary_masks):
#         current_image = np.zeros_like(tif_image)  
#         current_image[binary_mask] = tif_image[binary_mask]

#         current_image_with_trees, current_coords_of_maximums = show_trees2(current_image, ndvi_treshold=ndvi_treshold[numero_of_the_class] , elevation_threshold_min=elevation_threshold_min[numero_of_the_class], elevation_threshold_max=elevation_threshold_max[numero_of_the_class], maximum_filter_size=maximum_filter_size[numero_of_the_class], minimum_CC_size=minimum_CC_size[numero_of_the_class], output_name = output_name)

#         tab_coords_of_maximums.append(current_coords_of_maximums)


#     valid_coords_of_maximums = []
#     tab_numero_of_the_class = []
#     for numero_of_the_class, coords_of_maximums in enumerate(tab_coords_of_maximums):
#         for coord in coords_of_maximums:
#             if(len(valid_coords_of_maximums)==0):
#                 valid_coords_of_maximums.append(coord)
#                 tab_numero_of_the_class.append(numero_of_the_class)

#             else:
#                 for valid_coord in valid_coords_of_maximums:
#                     booleanFarEnough = True
#                     if(not isFarEnough(valid_coord, coord, 10)):
#                         booleanFarEnough = False
#                         break
#                 if(booleanFarEnough):
#                     valid_coords_of_maximums.append(coord)
#                     tab_numero_of_the_class.append(numero_of_the_class)

#     colors = [
#         [0, 255, 0],
#         [255, 0, 0],
#         [0, 0, 255],
#         [255, 255, 0],
#         [0, 255, 255],
#         [255, 0, 255],
#         [128, 128, 128]
#     ]


#     square_size=3       
#     final_image_with_trees = np.copy(tif_image)  
#     for numero_of_coord, coord in enumerate(valid_coords_of_maximums):
#         for i in range(coord[0] - square_size, coord[0] + square_size + 1):
#             for j in range(coord[1] - square_size, coord[1] + square_size + 1):
#                 if i >= 0 and i < tif_image.shape[0] and j >= 0 and j < tif_image.shape[1]:
#                     final_image_with_trees[i, j, :][:3] = colors[tab_numero_of_the_class[numero_of_coord]]  # Set pixel color to red

        



#     plt.figure(figsize=(20, 5))   
#     plt.imshow(final_image_with_trees[:,:,:3])
#     plt.title('Image with Trees Points and Contours')
#     for numero_of_the_class in range(len(binary_masks)):
#         if(numero_of_the_class == 2):
#             contours = find_contours(binary_masks[numero_of_the_class], 0.1)
#             for contour in contours:
#                 plt.plot(contour[:, 1], contour[:, 0], color=(0/255, 0/255, 255/255), linewidth=1)

#         if(numero_of_the_class == 4):
#             contours = find_contours(binary_masks[numero_of_the_class], 0.1)
#             for contour in contours:
#                 plt.plot(contour[:, 1], contour[:, 0], color=(0/255, 255/255, 0/255), linewidth=1)
#     plt.show()







#     plt.figure(figsize=(20, 5))
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [4, 1]})


#     ax1.imshow(final_image_with_trees[..., :3], cmap='viridis') 
#     ax1.axis('off')


#     class_names = ['coniferous', 'deciduous', 'brushwood', 'vineyard', 'herbaceous vegetation', 'ligneous', 'other']

#     for color, label in zip(colors, class_names):
#         color_normalized = np.array(color) / 255.0
#         ax2.scatter([], [], color=color_normalized, label=label, s=100)

#     ax2.axis('off')

#     ax2.legend(loc='center', bbox_to_anchor=(0, 0.5))
    
#     plt.show()






def show_trees_from_mask2(tif_image, old_mask, ndvi_treshold, elevation_threshold_min, elevation_threshold_max, maximum_filter_size, minimum_CC_size, output_name):
    new_mask = transform_to_only_trees_mask(old_mask)
    binary_masks = binary_masks_from_mask(new_mask, smoothing=1)

    tab_coords_of_maximums = []
    final_image_with_trees = np.copy(tif_image)
    for numero_of_the_class, binary_mask in enumerate(binary_masks):
        current_image = np.zeros_like(tif_image)  
        current_image[binary_mask] = tif_image[binary_mask]

        current_image_with_trees, current_coords_of_maximums = show_trees2(current_image, ndvi_treshold=ndvi_treshold[numero_of_the_class] , elevation_threshold_min=elevation_threshold_min[numero_of_the_class], elevation_threshold_max=elevation_threshold_max[numero_of_the_class], maximum_filter_size=maximum_filter_size[numero_of_the_class], minimum_CC_size=minimum_CC_size[numero_of_the_class], output_name = output_name)

        tab_coords_of_maximums.append(current_coords_of_maximums)


    valid_coords_of_maximums = []
    tab_numero_of_the_class = []
    for numero_of_the_class, coords_of_maximums in enumerate(tab_coords_of_maximums):
        for coord in coords_of_maximums:
            if(len(valid_coords_of_maximums)==0):
                valid_coords_of_maximums.append(coord)
                tab_numero_of_the_class.append(numero_of_the_class)

            else:
                for valid_coord in valid_coords_of_maximums:
                    booleanFarEnough = True
                    if(not isFarEnough(valid_coord, coord, 10)):
                        booleanFarEnough = False
                        break
                if(booleanFarEnough):
                    valid_coords_of_maximums.append(coord)
                    tab_numero_of_the_class.append(numero_of_the_class)

    colors = [
        [0, 255, 0],
        [255, 0, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [128, 128, 128]
    ]


    square_size=3       
    final_image_with_trees = np.copy(tif_image)  
    for numero_of_coord, coord in enumerate(valid_coords_of_maximums):
        for i in range(coord[0] - square_size, coord[0] + square_size + 1):
            for j in range(coord[1] - square_size, coord[1] + square_size + 1):
                if i >= 0 and i < tif_image.shape[0] and j >= 0 and j < tif_image.shape[1]:
                    final_image_with_trees[i, j, :][:3] = colors[tab_numero_of_the_class[numero_of_coord]]  # Set pixel color to red

        

    plt.figure(figsize=(20, 5))   
    plt.imshow(final_image_with_trees[:,:,:3])
    plt.title('Image with Trees Points and Contours')
    for numero_of_the_class in range(len(binary_masks)):
        if(numero_of_the_class == 2):
            contours = find_contours(binary_masks[numero_of_the_class], 0.1)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], color=(0/255, 0/255, 255/255), linewidth=1)

        if(numero_of_the_class == 4):
            contours = find_contours(binary_masks[numero_of_the_class], 0.1)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], color=(0/255, 255/255, 0/255), linewidth=1)
    plt.show()







    plt.figure(figsize=(20, 5))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [4, 1]})


    ax1.imshow(final_image_with_trees[..., :3], cmap='viridis') 
    ax1.axis('off')


    class_names = ['coniferous', 'deciduous', 'brushwood', 'vineyard', 'herbaceous vegetation', 'ligneous', 'other']

    for color, label in zip(colors, class_names):
        color_normalized = np.array(color) / 255.0
        ax2.scatter([], [], color=color_normalized, label=label, s=100)

    ax2.axis('off')

    ax2.legend(loc='center', bbox_to_anchor=(0, 0.5))
    
    plt.show()






def show_trees_from_mask4(tif_image, old_mask, ndvi_treshold, elevation_threshold_min, elevation_threshold_max, maximum_filter_size, minimum_CC_size, output_name):
    # new_mask = transform_to_only_trees_mask(old_mask)
    new_mask = old_mask
    binary_masks = binary_masks_from_mask_train(new_mask, smoothing=1)

    tab_coords_of_maximums = []
    final_image_with_trees = np.copy(tif_image)
    for numero_of_the_class, binary_mask in enumerate(binary_masks):
        current_image = np.zeros_like(tif_image)  
        current_image[binary_mask] = tif_image[binary_mask]

        current_image_with_trees, current_coords_of_maximums = show_trees2(current_image, ndvi_treshold=ndvi_treshold[numero_of_the_class] , elevation_threshold_min=elevation_threshold_min[numero_of_the_class], elevation_threshold_max=elevation_threshold_max[numero_of_the_class], maximum_filter_size=maximum_filter_size[numero_of_the_class], minimum_CC_size=minimum_CC_size[numero_of_the_class], output_name = output_name)

        tab_coords_of_maximums.append(current_coords_of_maximums)


    valid_coords_of_maximums = []
    tab_numero_of_the_class = []
    for numero_of_the_class, coords_of_maximums in enumerate(tab_coords_of_maximums):
        for coord in coords_of_maximums:
            if(len(valid_coords_of_maximums)==0):
                valid_coords_of_maximums.append(coord)
                tab_numero_of_the_class.append(numero_of_the_class)

            else:
                for valid_coord in valid_coords_of_maximums:
                    booleanFarEnough = True
                    if(not isFarEnough(valid_coord, coord, 10)):
                        booleanFarEnough = False
                        break
                if(booleanFarEnough):
                    valid_coords_of_maximums.append(coord)
                    tab_numero_of_the_class.append(numero_of_the_class)

    colors = [
        [0, 255, 0],
        [255, 0, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [128, 128, 128]
    ]


    square_size=3       
    final_image_with_trees = np.copy(tif_image)  
    for numero_of_coord, coord in enumerate(valid_coords_of_maximums):
        for i in range(coord[0] - square_size, coord[0] + square_size + 1):
            for j in range(coord[1] - square_size, coord[1] + square_size + 1):
                if i >= 0 and i < tif_image.shape[0] and j >= 0 and j < tif_image.shape[1]:
                    final_image_with_trees[i, j, :][:3] = colors[tab_numero_of_the_class[numero_of_coord]]  # Set pixel color to red

        

    plt.figure(figsize=(20, 5))   
    plt.imshow(final_image_with_trees[:,:,:3])
    plt.title('Image with Trees Points and Contours')
    for numero_of_the_class in range(len(binary_masks)):
        if(numero_of_the_class == 2):
            contours = find_contours(binary_masks[numero_of_the_class], 0.1)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], color=(0/255, 0/255, 255/255), linewidth=1)

        if(numero_of_the_class == 4):
            contours = find_contours(binary_masks[numero_of_the_class], 0.1)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], color=(0/255, 255/255, 0/255), linewidth=1)
    plt.show()





    plt.figure(figsize=(20, 5))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [4, 1]})


    ax1.imshow(final_image_with_trees[..., :3], cmap='viridis') 
    ax1.axis('off')


    class_names = ['coniferous', 'deciduous', 'brushwood', 'vineyard', 'herbaceous vegetation', 'ligneous', 'other']

    for color, label in zip(colors, class_names):
        color_normalized = np.array(color) / 255.0
        ax2.scatter([], [], color=color_normalized, label=label, s=100)

    ax2.axis('off')

    ax2.legend(loc='center', bbox_to_anchor=(0, 0.5))
    
    plt.show()