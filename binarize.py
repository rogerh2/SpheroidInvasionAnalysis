# Importing required modules for GUI and image processing
from tkinter import Tk, Frame, Button, Label, Entry, filedialog, messagebox, Scale, HORIZONTAL
from tkinter import Canvas, PhotoImage, Toplevel, Checkbutton, IntVar
from matplotlib import pyplot as plt
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
import os

# Define the BinarizedImage class with required functionalities
class BinarizedImage:

    def __init__(self, raw_image_path, save_path, threshold=36):
        self.img_path = raw_image_path
        self.save_fldr_path = save_path
        # Load the raw image (assuming it's grayscale)
        self.grayscale_array = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        self.binary_array = self.grayscale_array >= int(threshold)
        self.threshold = threshold
        self.contours = []

    def update_mask(self, threshold, boundary=None):
        """Update the mask based on a threshold and an optional boundary."""

        if boundary is not None:
            # Create a mask for the region inside the boundary
            mask = np.zeros_like(self.grayscale_array, dtype=bool).astype(np.uint8)
            cv2.fillPoly(mask, [np.array(boundary)], True)
            mask = mask.astype(bool)
            # Update the threshold only inside the boundary
            self.binary_array[mask] = self.grayscale_array[mask] >= int(threshold)
        else:
            # Update the threshold for the entire image
            self.threshold = threshold
            self.binary_array = self.grayscale_array >= int(threshold)

    def auto_contour(self, guassian_kernel=None):
        # If a Gaussian kernel is provided, apply Gaussian blur to the grayscale image.
        if guassian_kernel is not None:
            blurred_image = cv2.GaussianBlur(255 * self.binary_array.astype(np.uint8), guassian_kernel, 0)
            # Binarize the blurred image using the same threshold
            binary_blurred = blurred_image >= int(self.threshold)
            contours, _ = cv2.findContours(255 * binary_blurred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # Binarize the grayscale image using the current threshold
            binary_image = self.binary_array
            contours, _ = cv2.findContours(255 * binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        self.contours = contours


# Define the GUI class
class ImageBinarizationApp:
    def __init__(self, master):
        self.master = master
        master.title('Image Binarization App')

        # Main frame
        self.frame = Frame(master)
        self.frame.pack(fill='both', expand=True)

        # Add Main Menu buttons
        self.binarize_button = Button(self.frame, text="Binarize", command=self.open_binarize_window)
        self.binarize_button.pack()  # You will need to adjust the positioning according to your layout

        self.process_button = Button(self.frame, text="Process (Dummy)", command=self.dummy_process)
        self.process_button.pack()  # You will need to adjust the positioning according to your layout

        # Bind the resize event
        self.master.bind('<Configure>', self.resize_image_canvas)

        # Image display labels
        self.original_image_label = None
        self.binarized_image_label = None

        # Paths
        self.image_folder_path = ""
        self.save_folder_path = ""

        # Image objects
        self.current_image = None
        self.image_list = []
        self.image_index = 0

        # Threshold
        self.threshold_scale = None
        self.local_threshold = 36
        # Member variable to store the points for local threshold or deletion
        self.points = []

        # Zoom and pan settings
        self.zoom_scale = 1
        self.pan_start_x = 0
        self.pan_start_y = 0

        # Initialize pan offsets
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        # Initialize binarize window variables
        self.binarize_window = None
        self.left_canvas = None
        self.right_canvas = None
        self.threshold_scale = None
        self.zoom_in_button = None
        self.zoom_out_button = None

        self.drawing = False
        self.image_frame = None

        # Additional GUI components, initialized as None
        self.modify_button = None
        self.prev_button = None
        self.next_button = None
        self.save_button = None
        self.modify_pane = None
        self.local_thresh_scale = None
        self.blur_scale = None
        self.boundary_var = None
        self.boundary_check = None
        self.apply_button = None
        self.draw_var = None
        self.draw_check = None


    def dummy_process(self):
        messagebox.showinfo("Info", "This feature is not implemented yet.")

    def resize_image_canvas(self, event):
        # Calculate the new size while maintaining the aspect ratio
        if self.current_image and self.binarize_window.winfo_exists():
            self.update_canvas()

    def open_folder_dialog(self, folder_type):
        folder_selected = filedialog.askdirectory()
        if folder_type == "load":
            self.image_folder_path = folder_selected
            self.image_list = [f for f in os.listdir(self.image_folder_path) if f.endswith('.tif')]
        elif folder_type == "save":
            self.save_folder_path = folder_selected

    def load_image(self, image_path):
        # Load and display the image
        self.current_image = BinarizedImage(image_path, self.save_folder_path)

    def save_binarized_image(self):
        # Save the binarized image with and without the mask
        # ... (save image logic)
        pass

    def update_threshold(self, val):
        # Update the global threshold
        threshold_value = int(val)
        self.current_image.update_mask(threshold_value)
        self.update_canvas()

    def start_drawing(self, event):
        # Get the width and height of the right canvas
        canvas_width = self.right_canvas.winfo_width()
        canvas_height = self.right_canvas.winfo_height()

        # Check if the event happened within the canvas boundaries
        if 0 <= event.x < canvas_width and 0 <= event.y < canvas_height:
            # Clear previous points
            self.points = []
            # Start drawing
            self.drawing = True
            # Add the starting point
            self.add_point(event)

    def stop_drawing(self, event):
        # Stop drawing
        self.drawing = False
        # Update the state of the Local Threshold slider
        self.update_local_threshold_slider()

    def add_point(self, event):
        # Calculate the relative position of the mouse to the image
        canvas_width = int(self.zoom_scale * self.right_canvas.winfo_width())
        canvas_height = int(self.zoom_scale * self.right_canvas.winfo_height())
        img_width, img_height = self.current_image.grayscale_array.shape[::-1]

        # Adjust the event coordinates for pan offset and zoom scale
        adjusted_x = event.x - self.pan_offset_x
        adjusted_y = event.y - self.pan_offset_y

        # Scale the adjusted coordinates to the image dimensions
        x_scaled = int(adjusted_x * img_width / canvas_width)
        y_scaled = int(adjusted_y * img_height / canvas_height)

        # Add the scaled point to the points list
        self.points.append((x_scaled, y_scaled))

        # Visual feedback for the point
        self.right_canvas.create_oval(adjusted_x - 2, adjusted_y - 2, adjusted_x + 2
                                      , adjusted_y + 2, fill='red')

    def draw_boundary(self, event):
        # Continue adding points while drawing
        if self.drawing:
            self.add_point(event)

    def apply_local_threshold(self):
        # Check if there are enough points to define a boundary
        if len(self.points) > 2:
            # Apply the local threshold to the selected area
            self.current_image.update_mask(self.local_threshold, boundary=self.points)
            self.update_canvas()
            # Clear the points after applying the local threshold
            self.points = []
        else:
            messagebox.showinfo("Info", "Please draw a boundary on the image to apply a local threshold.")

    def delete_region(self):
        # Check if there are enough points to define a boundary
        if len(self.points) > 2:
            # Create a mask for the region inside the boundary and set to False
            mask = np.zeros_like(self.current_image.grayscale_array, dtype=bool)
            cv2.fillPoly(mask, [np.array(self.points)], True)
            self.current_image.binary_array[mask] = False
            self.update_canvas()
            # Clear the points after deletion
            self.points = []
        else:
            messagebox.showinfo("Info", "Please draw a boundary on the image to delete a region.")

    def navigate_images(self, direction):
        # Navigate to the next or previous image based on the direction
        if direction == 'next' and self.image_index < len(self.image_list) - 1:
            self.image_index += 1
        elif direction == 'prev' and self.image_index > 0:
            self.image_index -= 1

        # Load the image and update the canvas
        self.load_image(os.path.join(self.image_folder_path, self.image_list[self.image_index]))
        self.update_canvas()

    def skip_to_image(self, image_number):
        # Skip to a specific image number
        if 0 <= image_number < len(self.image_list):
            self.image_index = image_number
            self.load_image(os.path.join(self.image_folder_path, self.image_list[self.image_index]))
            self.update_canvas()
        else:
            messagebox.showinfo("Info", "Invalid image number.")

    def update_canvas(self):
        canvas_w = int(self.zoom_scale * self.left_canvas.winfo_width())
        canvas_h = int(self.zoom_scale * self.left_canvas.winfo_height())

        # Original image dimensions
        img_w, img_h = self.current_image.grayscale_array.shape[::-1]

        # Scaling factors
        scale_x = canvas_w / img_w
        scale_y = canvas_h / img_h

        # Resize the original image and update the left canvas
        original_image = Image.fromarray(self.current_image.grayscale_array)
        # Use Image.Resampling.LANCZOS for best downsampling quality
        original_photo = ImageTk.PhotoImage(original_image.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS))
        self.left_canvas.create_image(0, 0, anchor="nw", image=original_photo)
        self.left_canvas.image = original_photo  # Keep a reference to avoid garbage collection

        # Resize the binarized image and update the right canvas
        binarized_image = Image.fromarray((self.current_image.binary_array * 255).astype(np.uint8))
        # Use Image.Resampling.LANCZOS for best downsampling quality
        binarized_photo = ImageTk.PhotoImage(binarized_image.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS))
        self.right_canvas.create_image(0, 0, anchor="nw", image=binarized_photo)
        self.right_canvas.image = binarized_photo  # Keep a reference to avoid garbage collection

        # Draw contours if the boundary checkbox is checked
        if self.boundary_var:
            if self.boundary_var.get():
                # Determine the Gaussian kernel size from the blur scale or set to None if blur is 0
                kernel_size = None if self.blur_scale.get() == 0 else (self.blur_scale.get(), self.blur_scale.get())
                # Generate contours
                self.current_image.auto_contour(guassian_kernel=kernel_size)

                # Draw contours on both images
                for img_canvas, image_array in [(self.left_canvas, self.current_image.grayscale_array),
                                                (self.right_canvas, 255 * self.current_image.binary_array)]:
                    # Convert the image to BGR format for color drawing
                    contour_image = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    contour_image = cv2.resize(contour_image, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)

                    # Scale and shift contours
                    scaled_contours = [np.int32(np.array(contour) * [scale_x, scale_y]) for contour in
                                       self.current_image.contours]

                    # Draw the largest contour in green and others in red
                    cv2.drawContours(contour_image, scaled_contours, 0, (0, 255, 0), 2)
                    for i in range(1, len(scaled_contours)):
                        cv2.drawContours(contour_image, scaled_contours, i, (255, 0, 0), 2)

                    # Convert back to a PhotoImage and display on the canvas
                    photo_image = ImageTk.PhotoImage(image=Image.fromarray(contour_image))
                    img_canvas.create_image(0, 0, anchor="nw", image=photo_image)
                    img_canvas.image = photo_image  # Keep a reference to avoid garbage collection

    def open_binarize_window(self):
        # Hide the main window
        self.master.withdraw()

        # Open dialogs for folder selection
        self.open_folder_dialog("load")
        self.open_folder_dialog("save")

        # Create the binarize window
        self.binarize_window = Toplevel()
        self.binarize_window.title("Binarize Image")
        self.binarize_window.protocol("WM_DELETE_WINDOW", self.on_close_binarize_window)  # Handle the close event

        # Bind the resize event to the binarize window
        self.binarize_window.bind('<Configure>', self.resize_image_canvas)

        # Create a frame to hold the image canvases and the buttons
        self.image_frame = Frame(self.binarize_window)
        self.image_frame.pack(side='top', fill='both', expand=True)

        # Create and pack the left canvas for the original image
        self.left_canvas = Canvas(self.image_frame, width=250, height=250, bg='white')
        self.left_canvas.pack(side='left', fill='both', expand=True)

        # Create and pack the right canvas for the binarized image
        self.right_canvas = Canvas(self.image_frame, width=250, height=250, bg='white')
        self.right_canvas.pack(side='right', fill='both', expand=True)

        # Initialize variables for drawing
        self.drawing = False  # Track whether the mouse is currently being held down for drawing

        # Bind the mouse click event to start drawing
        self.right_canvas.bind("<ButtonPress-1>", self.on_mouse_click)
        # Bind the mouse release event to stop drawing
        self.right_canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        # Bind the mouse movement to draw continuously
        self.right_canvas.bind("<B1-Motion>", self.on_mouse_move)

        # Create and pack the threshold slider
        self.threshold_scale = Scale(self.binarize_window, from_=0, to_=255, orient=HORIZONTAL, command=self.update_threshold)
        self.threshold_scale.set(int(self.local_threshold))  # Set the default position of the slider
        self.threshold_scale.pack(fill='x')

        # Add buttons for local thresholding, deleting regions, navigation, and saving
        button_frame = Frame(self.binarize_window)
        button_frame.pack(side='bottom', fill='x')

        # Modify button frame
        modify_frame = Frame(self.binarize_window)
        modify_frame.pack(side='right', fill='y')

        # Zoom In and Zoom Out buttons
        self.zoom_in_button = Button(self.image_frame, text="Zoom In", command=lambda: self.update_zoom(1.25))
        self.zoom_in_button.pack(side='left')

        self.zoom_out_button = Button(self.image_frame, text="Zoom Out", command=lambda: self.update_zoom(0.8))
        self.zoom_out_button.pack(side='left')

        # Modify button
        self.modify_button = Button(modify_frame, text="Modify", command=self.open_modify_pane)
        self.modify_button.pack()

        self.prev_button = Button(button_frame, text="<< Prev", command=lambda: self.navigate_images('prev'))
        self.prev_button.pack(side='left')

        self.next_button = Button(button_frame, text="Next >>", command=lambda: self.navigate_images('next'))
        self.next_button.pack(side='left')

        self.save_button = Button(button_frame, text="Save", command=self.save_binarized_image)
        self.save_button.pack(side='left')

        # If there are images in the folder, load the first one
        if self.image_list:
            self.load_image(os.path.join(self.image_folder_path, self.image_list[0]))
            self.update_canvas()

    def on_close_binarize_window(self):
        """Close the binarize window and reset related variables to their default settings."""
        if self.binarize_window:
            self.binarize_window.destroy()  # Destroy the binarize window
            self.binarize_window = None  # Reset the window variable

        # Reset all variables associated with the binarize window to their default values
        self.left_canvas = None
        self.right_canvas = None
        self.threshold_scale = None
        self.zoom_in_button = None
        self.zoom_out_button = None
        self.image_frame = None

        # Reset drawing-related variables
        self.drawing = False
        self.points = []

        # Reset navigation and image-related variables
        self.image_list = []
        self.image_index = 0
        self.current_image = None

        # Reset zoom and pan settings
        self.zoom_scale = 1
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        # Zoom In and Zoom Out buttons
        self.zoom_in_button = None
        self.zoom_out_button = None

        # Modify button
        self.modify_button = None
        self.prev_button = None
        self.next_button = None
        self.save_button = None

        # Close the modify pane if it's open
        self.on_close_modify_pane()

        # Show the main window again
        self.master.deiconify()

    def open_modify_pane(self):
        # Create the modify pane as a Toplevel window
        self.modify_pane = Toplevel(self.binarize_window)
        self.modify_pane.title("Modifications")
        self.modify_pane.protocol("WM_DELETE_WINDOW", self.on_close_modify_pane)  # Handle the close event

        # Local Threshold Slider
        self.local_thresh_scale = Scale(self.modify_pane, from_=0, to_=255, orient=HORIZONTAL, label="Local Threshold")
        self.local_thresh_scale.pack(fill='x')

        # Blur Slider
        self.blur_scale = Scale(self.modify_pane, from_=0, to_=10, orient=HORIZONTAL, label="Blur")
        self.blur_scale.pack(fill='x')

        # Boundary Checkbox
        self.boundary_var = IntVar()
        self.boundary_check = Checkbutton(self.modify_pane, text="Boundary", variable=self.boundary_var)
        self.boundary_check.pack()

        # Apply Button
        self.apply_button = Button(self.modify_pane, text="Apply", command=self.apply_modifications)
        self.apply_button.pack()

        # Draw Checkbox
        self.draw_var = IntVar()
        self.draw_check = Checkbutton(self.modify_pane, text="Draw", variable=self.draw_var)
        self.draw_check.pack()

        # Update the state of the Local Threshold slider based on the points
        self.update_local_threshold_slider()

    def on_close_modify_pane(self):
        """Close the modify pane and reset related variables to their default values."""
        if self.modify_pane:
            self.modify_pane.destroy()  # Destroy the modify pane window
            self.modify_pane = None  # Reset the window variable

        # Reset all variables associated with the modify pane to their default values
        self.local_thresh_scale = None
        self.blur_scale = None
        self.boundary_var = None
        self.draw_check = None
        self.apply_button = None
        self.draw_var = None

    def update_local_threshold_slider(self):
        if len(self.points) > 0:
            self.local_thresh_scale.config(state="normal")
        else:
            self.local_thresh_scale.config(state="disabled")

    def apply_modifications(self):
        # Apply the local_threshold
        if len(self.points) > 0:
            self.local_threshold = self.local_thresh_scale.get()
            self.apply_local_threshold()

        # Update the canvas with potential modifications
        self.update_canvas()

    def start_panning(self, event):
        if self.draw_var:
            if not self.draw_var.get():  # Only start panning if not in draw mode
                self.pan_start_x, self.pan_start_y = event.x, event.y
        else:
            self.pan_start_x, self.pan_start_y = event.x, event.y

    def stop_panning(self, event):
        if self.draw_var:
            if not self.draw_var.get():  # Only start panning if not in draw mode
                # Update pan offsets
                self.pan_offset_x += event.x - self.pan_start_x
                self.pan_offset_y += event.y - self.pan_start_y
        else:
            # Update pan offsets
            self.pan_offset_x += event.x - self.pan_start_x
            self.pan_offset_y += event.y - self.pan_start_y

    def pan_image(self, event):
        if self.draw_var:
            if not self.draw_var.get():  # Only pan if not in draw mode
                dx = event.x - self.pan_start_x
                dy = event.y - self.pan_start_y
                self.left_canvas.scan_dragto(dx + self.pan_offset_x, dy + self.pan_offset_y, gain=1)
                self.right_canvas.scan_dragto(dx + self.pan_offset_x, dy + self.pan_offset_y, gain=1)
                # self.pan_start_x, self.pan_start_y = event.x, event.y
        else:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y

            self.left_canvas.scan_dragto(dx + self.pan_offset_x, dy + self.pan_offset_y, gain=1)
            self.right_canvas.scan_dragto(dx + self.pan_offset_x, dy + self.pan_offset_y, gain=1)
            # self.pan_start_x, self.pan_start_y = event.x, event.y

    def update_zoom(self, zoom_factor):
        self.zoom_scale *= zoom_factor
        self.update_canvas()


    def on_mouse_click(self, event):
        # Decide whether to start drawing or panning based on the 'Draw' checkbox
        if self.draw_var:
            if self.draw_var.get():
                self.start_drawing(event)
            else:
                self.start_panning(event)
        else:
            self.start_panning(event)

    def on_mouse_move(self, event):
        # Decide whether to draw or pan based on the 'Draw' checkbox
        if self.draw_var:
            if self.draw_var.get():
                self.draw_boundary(event)
            else:
                self.pan_image(event)
        else:
            self.pan_image(event)

    def on_mouse_release(self, event):
        # Stop drawing or panning
        if self.draw_var:
            if self.draw_var.get():
                self.stop_drawing(event)
            else:
                self.stop_panning(event)
        else:
            self.stop_panning(event)


# ... (rest of the code, including the main() function remains unchanged)

def main():
    # Create the main window (root of the Tk interface)
    root = Tk()
    # Set the dimensions of the window
    root.geometry("800x600")

    # Create the application
    app = ImageBinarizationApp(root)

    # Start the application loop
    root.mainloop()


# Run the main application loop
if __name__ == '__main__':
    main()