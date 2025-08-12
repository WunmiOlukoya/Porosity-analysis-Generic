

# Import statements:

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import sys 
import os

#import custom module
from image_analysis import percentage_porosity_largePart

# Set up the window
window = tk.Tk()
window.title("Porosity calculator")
window.resizable(width=False, height=False)

def calculate_porosity():
    """
    This function is called when the 'Calculate' button is pressed.
    It gets the file path, calls the porosity function, and updates the result label.
    """
    file_path = ent_file.get()
    
    if not file_path:
        # Open a file dialog if the entry is empty
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif")]
        )
        if not file_path:
            return  # Exit if the user cancels the file dialog
        ent_file.delete(0, tk.END)
        ent_file.insert(0, file_path)
    
    lbl_result.config(text="Calculating...")
    result = percentage_porosity_largePart(file_path)
    lbl_result.config(text=f"Porosity: {result}")

# Create the file location entry frame with an Entry
# widget and label in it
frm_entry = tk.Frame(master=window)

# Enter the file location for the image 
# The width is set to a more reasonable size.
ent_file = tk.Entry(master=frm_entry, width=50)
lbl_image = tk.Label(master=frm_entry, text="Image File:")

# Layout the file Entry and Label in frm_entry
# using the .grid() geometry manager
lbl_image.grid(row=0, column=0, sticky="e")
ent_file.grid(row=0, column=1, sticky="w")

# Create the conversion Button and result display Label
# The command is now set to the new `calculate_porosity` function
btn_convert = tk.Button(
    master=window,
    text= "Calculate",
    command=calculate_porosity
)

lbl_result = tk.Label(master=window, text="Porosity result", fg="blue")

# Layout the main widgets in the window
frm_entry.grid(row=0, column=0, padx=10, pady=10)
btn_convert.grid(row=0, column=1, pady=10)
lbl_result.grid(row=1, column=0, columnspan=2, pady=10)

window.mainloop()