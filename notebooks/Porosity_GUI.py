
# main_script.py (Updated)
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image # Import Pillow for image handling
from image_analysis import percentage_porosity_largePart

# Set up the window
window = tk.Tk()
window.title("Porosity calculator")
window.resizable(width=False, height=False)

def calculate_porosity():
    file_path = ent_file.get()
    
    if not file_path:
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif")]
        )
        if not file_path:
            return
        ent_file.delete(0, tk.END)
        ent_file.insert(0, file_path)
    
    lbl_result.config(text="Calculating...")
    
    # Now the function returns two values
    porosity_result, processed_img = percentage_porosity_largePart(file_path)
    
    if processed_img:
        # Resize the image for display in the GUI
        img_width, img_height = processed_img.size
        new_width = 400
        new_height = int((new_width / img_width) * img_height)
        resized_img = processed_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert the Pillow image to a Tkinter-compatible format
        photo_img = ImageTk.PhotoImage(resized_img)
        
        # Update the label with the new image
        lbl_image_display.config(image=photo_img)
        lbl_image_display.image = photo_img # Keep a reference to prevent garbage collection
        
    lbl_result.config(text=f"Porosity: {porosity_result}")


# Create the GUI elements as before
frm_entry = tk.Frame(master=window)
ent_file = tk.Entry(master=frm_entry, width=50)
lbl_image_label = tk.Label(master=frm_entry, text="Image File:")
lbl_image_label.grid(row=0, column=0, sticky="e")
ent_file.grid(row=0, column=1, sticky="w")

btn_convert = tk.Button(
    master=window,
    text= "Calculate",
    command=calculate_porosity
)

lbl_result = tk.Label(master=window, fg="blue")

# New Label widget to display the image
lbl_image_display = tk.Label(master=window)

# Layout the main widgets in the window
frm_entry.grid(row=0, column=0, padx=10, pady=10)
btn_convert.grid(row=0, column=1, pady=10)
lbl_result.grid(row=1, column=0, columnspan=2, pady=10)
lbl_image_display.grid(row=2, column=0, columnspan=2, pady=10)

window.mainloop()