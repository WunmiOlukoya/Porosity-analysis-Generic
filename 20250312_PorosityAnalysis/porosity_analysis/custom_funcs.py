# custom_funcs.py

# modified from https://github.com/bioimagebook/bioimagebook.github.io/blob/main/helpers/__init__.py#L10

import matplotlib.pyplot as plt
import os 
import pandas as pd
from typing import Dict, List, Tuple, Union, IO, Iterable, Sequence
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import sys

DEFAULT_DPI = 200

### Make the plots pretty 
def scatter_plot_template():  
    fig,ax = plt.subplots(figsize=(10,6))
    plt.rc('font', size=20)  # Change all fonts
    # Plot appearance 
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='both' , linestyle='-',color='k' ,alpha = 0.2)
    plt.grid(visible=True, which='major', axis='both' , linestyle='-',color='k' ,alpha = 0.7)

    plt.minorticks_on() 

    return None

### Image analysis 
def load_image(file):
    try:
        im = iio.imread(file)
        return im 
    except Exception as err:
        print(err)
        print(f'Error reading image ,{file} does not exists')


def is_greyscale(image):

    image_tuple = len(image.shape)
    if image_tuple == 2:
        gs_image = image
        return gs_image

    elif image_tuple == 3: # check if image is greyscale 
        gs_image = image[:,:,0]
        return gs_image
        



def my_imshow(im, title=None, cmap='gray'):
    """
    Call imshow and turn the axis off, optionally setting a title and colormap.
    The default colormap is 'gray', and there is no default title.
    """

    # Show image & turn off axis
    plt.imshow(im, cmap=cmap)
    plt.axis(False)

    # Show a title if we have one
    if title is not None:
        plt.title(title)

    plt.show()


def create_figure(figsize=None, dpi=DEFAULT_DPI, **kwargs):
    return plt.figure(figsize=figsize, dpi=dpi, **kwargs)