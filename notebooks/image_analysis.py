# ==============================================================================
# ==============================================================================
# ============================ IMPORTING LIBRARIES =============================
# ==============================================================================
# ==============================================================================

# Image processing imports
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import label, regionprops, regionprops_table
from scipy.ndimage import binary_fill_holes
import imageio.v3 as iio
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from io import BytesIO
# ==============================================================================
# ==============================================================================
# ============================= START OF FUNCTIONS =============================
# ==============================================================================
# ==============================================================================


def percentage_porosity_largePart(file_name):
    """
    Calculates the percentage porosity of the largest object in an image.
    
    Args:
        file_name (str): The path to the image file.
        
    Returns:
        str: The calculated percentage porosity as a formatted string or an error message.
    """
    try:
        img = iio.imread(file_name)
    except Exception as e:
        return f"Error reading image: {e}"
        
    if img.ndim == 3:
        image_grayscale = rgb2gray(img)
    else:
        image_grayscale = img

    thresh = threshold_otsu(image_grayscale)
    binary = image_grayscale > thresh

    removed_small_objects = remove_small_objects(binary, min_size=6)
    tidy_image = remove_small_holes(removed_small_objects, area_threshold=5)

    label_image = label(tidy_image)
    regions = regionprops(label_image)

    if not regions:
        return "No objects found in the image mask."

    largest_region = max(regions, key=lambda r: r.area)
    largest_object_mask = (label_image == largest_region.label)
    
    filled_mask = binary_fill_holes(largest_object_mask)
    holes_mask = filled_mask != largest_object_mask
    labeled_holes = label(holes_mask)

    props = regionprops_table(labeled_holes, properties=['area'])
    data = pd.DataFrame(props)
    
    if data.empty:
        return "No pores found."
        
    sum_pores_px = data['area'].sum()
    Total_Area = np.count_nonzero(filled_mask)
    
    # Corrected porosity calculation to find the percentage of holes
    percentage_porosity = (1 - (sum_pores_px / Total_Area)) * 100

    # Create a simple figure to draw the original image and pores
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_title(f"Detected Pores (Porosity: {percentage_porosity:.2f}%)")
    
    # Overlay the pores in a contrasting color (e.g., red)
    ax.imshow(holes_mask, cmap='hot', alpha=0.5)
    ax.axis('off')
    
    # Save the figure to a buffer instead of a file
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig) # Close the figure to free memory
    buf.seek(0)
    
    # Open the image from the buffer with Pillow
    pil_image = Image.open(buf)
    
    return f"The part is {percentage_porosity:.2f}% dense" ,pil_image