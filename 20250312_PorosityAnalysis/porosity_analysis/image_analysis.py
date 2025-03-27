# ==============================================================================
# ==============================================================================
# ============================ IMPORTING LIBRARIES =============================
# ==============================================================================
# ==============================================================================


# modified from JF Croteau, Berkely labs

import pandas as pd # Data analysis 
import numpy as np # Generic library with matrices
import matplotlib.pyplot as plt # To plot anything
import seaborn as sns # To plot results (more nicely)

# Scikit-image
from skimage import data, filters, measure # All of the image analysis is from this

from skimage.io import imshow, imread, imsave
from skimage.morphology import remove_small_holes, remove_small_objects, binary_dilation, binary_erosion
from skimage.color import rgb2gray, label2rgb, rgb2hsv
from skimage.segmentation import flood, flood_fill, clear_border
from skimage.transform import rotate
from skimage.filters import threshold_multiotsu ,gaussian


# K-means
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle

#Image reading 
import os 
import imageio.v3 as iio
from pathlib import Path

# ==============================================================================
# ==============================================================================
# ============================= START OF FUNCTIONS =============================
# ==============================================================================
# ==============================================================================


# ===== Read and show original image =====
def read_image(dir, filename, bool_plot):
    image = imread(dir+filename)

    if image.ndim > 2:
        if np.shape(image)[2] == 4:
            image = np.delete(image, 0, 2)
        
    if bool_plot:
        plt.imshow(image)
    return(image)

# ===== Read and save image as .npy =====
def png_toNumpy(raw_image_dir):
    """
    Converts PNG images in a directory to .npy files and saves them in an 'npy_cache' subfolder.

    Args:
        raw_image_dir (str): Directory containing cropped PNG images.

    Returns:
        str: Path to the folder where .npy files are saved.
    """

    raw_image_dir = Path(raw_image_dir)  # Convert to Path object for easier handling
    npy_save_folder = raw_image_dir / 'npy_cache'  # Create a subdirectory for .npy files
    npy_save_folder.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists


    for img_png in raw_image_dir.iterdir():  # Iterate through all files in the directory
        if img_png.suffix.lower() == '.png' and 'lab' not in img_png.name:  # Check file extension & filter out labeled images
            
            try:
                print(f"Processing: {img_png}")
                img_np_array = np.array(iio.imread(img_png))  # Read image and convert to NumPy array

                save_location = npy_save_folder / f"{img_png.stem}.npy"  # Define output path
                
                if not save_location.exists():  # Save only if file doesn't exist
                    np.save(save_location, img_np_array)
                    print(f"Saved: {save_location}")
                else:
                    print(f"Skipped (already exists): {save_location}")

            except Exception as e:
                print(f"Error processing {img_png}: {e}")

    return str(npy_save_folder)
                
                


# ===== Plot data about and show images =====

def show_before_after(image_before, image_after, str_change, i_cmap):
    if i_cmap == 1:
        str_cmap = "gray"
    elif i_cmap == 2:
        str_cmap = "plasma"
    elif i_cmap == 3:
        str_cmap = "seismic"
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    axes[0].imshow(image_before, cmap=str_cmap)
    axes[0].set_title("Before "+str_change)
    axes[1].imshow(image_after, cmap=str_cmap)
    axes[1].set_title("After "+str_change)

def image_hist(image):
    image_1D = image.ravel()
    plt.figure(figsize=(12,4))
    sns.histplot(data=image_1D)
    plt.yscale('linear')

def image_hist_3ch(image):
    R = image[:,:,0].ravel()
    G = image[:,:,1].ravel()
    B = image[:,:,2].ravel()

    image_hist(R,'r','R',255)
    image_hist(G,'g','G',255)
    image_hist(B,'b','B',255)


# ===== Manipulate images =====

def crop_image(image_original,y1_px,y2_px, bool_plot):
    cropped = image_original[y1_px:y2_px,:] # removing any text/pixels at the bottom of the image, e.g SEM
    if bool_plot:
        show_before_after(image_original, cropped, "cropped", 1) # image_before, image_after, str_change, i_cmap
    return(cropped)

def grayscale_image(image_original, bool_plot):    
    # Convert to grayscale
    if image_original.ndim == 2:
        imag_grayscale = image_original
        print("image is already grayscale")
    else:
        imag_grayscale = rgb2gray(image_original)

    if bool_plot:
        show_before_after(image_original, imag_grayscale, "grayscale", 1) # image_before, image_after, str_change, i_cmap
    return(imag_grayscale)

def rotate_image(image_original, rot_deg, bool_plot):
    # Rotation
    imag_rot = rotate(image_original, rot_deg) # deg angle CCW
    if bool_plot:
        show_before_after(image_original, imag_rot, "rotated", 1) # image_before, image_after, str_change, i_cmap
    return(imag_rot)

def removed_4th_dimension(image_original):
    if np.shape(image_original)[2] == 4:
        imag_3D = np.delete(image_original, 3, 2) # Deleting last column
        print(np.shape(imag_3D)[2])



def filter_gaussian(image_original, n_sigma, bool_plot):
    image_filtered = gaussian(image_original, sigma=n_sigma)
    if bool_plot:
        show_before_after(image_original, image_filtered, "Gaussian filter", 1) # image_before, image_after, str_change, i_cmap
    return(image_filtered)



# ===== Masks and thresholds =====

def threshold_larger_than(image_original, n_threshold,bool_plot):
    image_mask = image_original > n_threshold # Every pixel with a value greater than n0 will be kept (TRUE)
    if bool_plot:
        show_before_after(image_original, image_mask, "threshold", 1) # image_before, image_after, str_change, i_cmap
    return(image_mask) # Boolean

def threshold_smaller_than(image_original, n_threshold,bool_plot):
    image_mask = image_original < n_threshold # Every pixel with a value greater than n0 will be kept (TRUE)
    if bool_plot:
        show_before_after(image_original, image_mask, "threshold", 1) # image_before, image_after, str_change, i_cmap
    return(image_mask) # Boolean

def clean_small_objects(image_original,n_objects,bool_plot):
    image_mask = remove_small_objects(image_original, min_size=n_objects)
    if bool_plot:
        show_before_after(image_original, image_mask, "small objects", 1) # image_before, image_after, str_change, i_cmap
    return(image_mask) # Boolean

def clean_large_objects(image_original,n_objects,bool_plot):
    image_mask = remove_small_objects(image_original, min_size=n_objects)
    image_mask = np.logical_and(image_original,np.logical_not(image_mask))
    if bool_plot:
        show_before_after(image_original, image_mask, "small objects", 1) # image_before, image_after, str_change, i_cmap
    return(image_mask) # Boolean

def clean_small_holes(image_original,n_holes,bool_plot):
    image_mask = remove_small_holes(image_original, area_threshold=n_holes)
    if bool_plot:
        show_before_after(image_original, image_mask, "small holes", 1) # image_before, image_after, str_change, i_cmap
    return(image_mask) # Boolean

def erosion_dilation_loop(image_original, n_erosion, n_dilation,bool_plot):
    image_eroded = image_original
    for i in range(n_erosion):
        image_eroded = binary_erosion(image_eroded)
    image_dilated = image_eroded
    for i in range(n_dilation):
        image_dilated = binary_dilation(image_dilated)
    
    if bool_plot:
        fig,axes = plt.subplots(1,3,figsize=(18,4))
        axes[0].imshow(image_original)
        axes[0].set_title("Original")
        axes[1].imshow(image_eroded)
        axes[1].set_title("After erosion")
        axes[2].imshow(image_dilated)
        axes[2].set_title("After dilation")

    return(image_dilated)

def multi_otsu(image, n_classes):
    image_otsu = threshold_multiotsu(image, classes=n_classes)
    regions = np.digitize(image, bins=image_otsu)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 4))

    # Plotting the original image.
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    ax[1].hist(image.ravel(), bins=255)
    ax[1].set_title('Histogram')
    for thresh in image_otsu:
        ax[1].axvline(thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(regions, cmap='jet')
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')

# Unsupervised machine learning
def smp_kmeans(image_original, n_clus, bool_plot):
    if image_original.ndim == 3:
        w,h,d = np.shape(image_original)
        kmeans = KMeans(n_clusters=n_clus,n_init=50,max_iter=100).fit(np.reshape(image_original, (w * h, d)))
        labels = kmeans.predict(np.reshape(image_original, (w * h, d)))
    elif image_original.ndim == 2:
        w,h = np.shape(image_original)
        kmeans = KMeans(n_clusters=n_clus,n_init=50,max_iter=100).fit(np.reshape(image_original, (w * h,1)))
        labels = kmeans.predict(np.reshape(image_original, (w * h,1)))
    image_kmeans = np.reshape(labels, (w, h))
    if bool_plot:
        show_before_after(image_original, image_kmeans, "kmeans clustering", 2) # image_before, image_after, str_change, i_cmap
    return(image_kmeans)

# ===== Labeling and segmentation =====

def label_segment(image_original,bool_plot):
    label = measure.label(image_original)
    label_rgb = label2rgb(label)
    if bool_plot:
        show_before_after(image_original, label_rgb, "segmentation", 1) # image_before, image_after, str_change, i_cmap
    return(label)


# ===== Quantitative analyses =====

def filament_properties(label_matrix,centroid_wire):
    props = measure.regionprops_table(label_matrix, properties=('label','area','centroid','orientation','axis_major_length','axis_minor_length','eccentricity','solidity'))
    df_props = pd.DataFrame(props)
    df_props['AR'] = np.divide(df_props['axis_major_length'],df_props['axis_minor_length'])
    df_props['distance_center'] = np.sqrt((centroid_wire[0]-df_props['centroid-0'].values)**2 + (centroid_wire[1]-df_props['centroid-1'].values)**2)
    # df_props.drop(0,0,inplace=True)
    return(df_props)

def wire_properties(label_matrix):
    props = measure.regionprops_table(label_matrix, properties=('label','area','centroid','orientation','axis_major_length','axis_minor_length','eccentricity'))
    df_props = pd.DataFrame(props)
    df_props['AR'] = np.divide(df_props['axis_major_length'],df_props['axis_minor_length'])
    return(df_props)

def label_quantitative(label_matrix_def,df_props_def):
    area_label = np.zeros([np.shape(label_matrix_def)[0],np.shape(label_matrix_def)[1]])
    AR_label = np.zeros([np.shape(label_matrix_def)[0],np.shape(label_matrix_def)[1]])
    eccentricity_label = np.zeros([np.shape(label_matrix_def)[0],np.shape(label_matrix_def)[1]])
    distance_label = np.zeros([np.shape(label_matrix_def)[0],np.shape(label_matrix_def)[1]])
    for i_color in range(np.shape(label_matrix_def)[0]):
        for j_color in range(np.shape(label_matrix_def)[1]):
            if label_matrix_def[i_color][j_color] == 0:
                area_label[i_color][j_color] = 0
                AR_label[i_color][j_color] = 0
                eccentricity_label[i_color][j_color] = 0
                distance_label[i_color][j_color] = 0
            else:
                area_label[i_color][j_color] = df_props_def['area'][label_matrix_def[i_color][j_color]-1]
                AR_label[i_color][j_color] = df_props_def['AR'][label_matrix_def[i_color][j_color]-1]
                eccentricity_label[i_color][j_color] = df_props_def['solidity'][label_matrix_def[i_color][j_color]-1]
                distance_label[i_color][j_color] = df_props_def['distance_center'][label_matrix_def[i_color][j_color]-1]
    return(area_label,AR_label,eccentricity_label,distance_label)