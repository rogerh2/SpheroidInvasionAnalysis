# Importing required modules for GUI and image processing
from tkinter import Canvas, Toplevel, Checkbutton, IntVar, Tk, Frame, Button, Label, Entry, filedialog\
    , messagebox, Scale, HORIZONTAL, LEFT, TOP, X, ttk, NORMAL, DISABLED
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import pandas as pd
import threading

from binarize import BinarizedImage
from quantify_image_set import analysis_logic


class MainMenu:

    def __init__(self, master):
        self.master = master
        self.binarize_ap = ImageBinarizationApp(master)
        self.analyze_ap = SpheroidAnalysisApp(master)

        master.title('Image Binarization App')

        # Main frame
        self.frame = Frame(master)
        self.frame.pack(fill='both', expand=True)

        # Add Main Menu buttons
        self.binarize_button = Button(self.frame, text="Binarize", command=self.binarize_ap.open_folder_selection_popup)
        self.binarize_button.pack()  # You will need to adjust the positioning according to your layout

        self.process_button = Button(self.frame, text="Analyize", command=self.analyze_ap.open_analyze_window)
        self.process_button.pack()  # You will need to adjust the positioning according to your layout

        # Bind the resize event
        self.master.bind('<Configure>', self.binarize_ap.resize_image_canvas)


# Define the GUI class
class ImageBinarizationApp:
    def __init__(self, master):
        self.master = master

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
        self.oval_ids = []

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
        self.popup = None
        self.continue_button = None
        self.set_scale_loc_only = False

        # In update_canvas method
        self.left_image_id = None
        self.right_image_id = None

        self.recent_threshold = self.local_threshold  # Initialize with the default local threshold
        self.image_states = []
        self.threshold_states = []
        self.current_state_index = -1

    def save_state(self):
        # Save the current binary array as a state
        self.image_states = self.image_states[:self.current_state_index + 1]  # Truncate the redo history
        self.threshold_states = self.threshold_states[:self.current_state_index + 1]

        self.image_states.append(np.copy(self.current_image.binary_array))
        self.threshold_states.append(np.copy(self.recent_threshold))
        self.current_state_index += 1

    def restore_state(self, state_index):
        # Restore a saved state
        if 0 <= state_index < len(self.image_states):
            self.delete_points()
            self.set_scale_loc_only = True
            self.recent_threshold = self.threshold_states[state_index]
            self.local_thresh_scale.set(np.copy(self.recent_threshold))
            self.current_image.binary_array = np.copy(self.image_states[state_index])
            self.update_canvas()
            self.current_state_index = state_index

    def resize_image_canvas(self, event):
        # Calculate the new size while maintaining the aspect ratio
        if self.current_image and self.binarize_window.winfo_exists():
            self.update_canvas()

    def load_image(self, image_path):
        # Load and display the image
        self.current_image = BinarizedImage(image_path, self.save_folder_path)
        self.update_threshold(self.recent_threshold)  # Apply the most recent threshold

        # Save the initial image and reset the image state history
        self.image_states = []
        self.current_state_index = -1
        self.save_state()

    def save_binarized_image(self):
        self.current_image.save_binarized_image()

    def update_threshold(self, val):
        threshold_value = int(val)

        if self.set_scale_loc_only:
            self.set_scale_loc_only = False
            return

        # Check if there are enough points to define a boundary, if so update local threshold, otherwise update global
        if len(self.points) > 2:
            # Apply the local threshold to the selected area
            self.current_image.update_mask(threshold_value, boundary=self.points)
            self.update_canvas()

        else:
            # Update the global threshold
            self.local_threshold = threshold_value
            self.current_image.update_mask(threshold_value)
            self.update_canvas()

    def start_drawing(self, event):
        # Get the width and height of the right canvas
        canvas_width = self.right_canvas.winfo_width()
        canvas_height = self.right_canvas.winfo_height()

        # Check if the event happened within the canvas boundaries
        if 0 <= event.x < canvas_width and 0 <= event.y < canvas_height:
            # Clear previous points
            self.delete_points()
            # Start drawing
            self.drawing = True
            # Add the starting point
            self.add_point(event)

    def stop_drawing(self, event):
        # Stop drawing
        self.drawing = False

    def add_point(self, event):
        # Calculate the relative position of the mouse to the image
        canvas_width = int(self.zoom_scale * self.right_canvas.winfo_width())
        canvas_height = int(self.zoom_scale * self.right_canvas.winfo_height())
        img_width, img_height = self.current_image.grayscale_array.shape[::-1]

        # Translate window coordinates to canvas coordinates
        canvas_x = self.right_canvas.canvasx(event.x)
        canvas_y = self.right_canvas.canvasy(event.y)

        # Adjust the event coordinates for pan offset and zoom scale
        adjusted_x = canvas_x
        adjusted_y = canvas_y

        # Scale the adjusted coordinates to the image dimensions
        x_scaled = int(adjusted_x * img_width / canvas_width)
        y_scaled = int(adjusted_y * img_height / canvas_height)

        # Add the scaled point to the points list
        self.points.append((x_scaled, y_scaled))

        # Visual feedback for the point
        oval_id = self.right_canvas.create_oval(adjusted_x - 2, adjusted_y - 2, adjusted_x + 2
                                      , adjusted_y + 2, fill='red')
        self.oval_ids.append(oval_id)  # Append the ID to the list

    def draw_boundary(self, event):
        # Continue adding points while drawing
        if self.drawing:
            self.add_point(event)

    def delete_points(self):
        for oval_id in self.oval_ids:
            self.right_canvas.delete(oval_id)
        self.oval_ids = []  # Reset the list after deleting the ovals
        self.points = []

    def delete_region(self):
        # Check if there are enough points to define a boundary
        if len(self.points) > 2:
            # Create a mask for the region inside the boundary and set to False
            mask = np.zeros_like(self.current_image.grayscale_array, dtype=bool)
            cv2.fillPoly(mask, [np.array(self.points)], True)
            self.current_image.binary_array[mask] = False
            self.update_canvas()
            # Clear the points after deletion
            self.delete_points()
        else:
            messagebox.showinfo("Info", "Please draw a boundary on the image to delete a region.")

    def navigate_images(self, direction):
        # Navigate to the next or previous image based on the direction
        if direction == 'next' and self.image_index < len(self.image_list) - 1:
            self.save_binarized_image()
            self.image_index += 1
        elif direction == 'prev' and self.image_index > 0:
            self.image_index -= 1

        # Load the image and update the canvas
        self.load_image(os.path.join(self.image_folder_path, self.image_list[self.image_index]))
        self.update_canvas()

        # Bring the modify pane to front if it's open
        if self.modify_pane and self.modify_pane.winfo_exists():
            self.modify_pane.lift()

    def skip_to_image(self, image_number):
        # Skip to a specific image number
        if 0 <= image_number < len(self.image_list):
            self.image_index = image_number
            self.load_image(os.path.join(self.image_folder_path, self.image_list[self.image_index]))
            self.update_canvas()
        else:
            messagebox.showinfo("Info", "Invalid image number.")

    def update_canvas(self, *args):
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
        self.left_image_id = self.left_canvas.create_image(0, 0, anchor="nw", image=original_photo)
        self.left_canvas.image = original_photo  # Keep a reference to avoid garbage collection

        # Resize the binarized image and update the right canvas
        binarized_image = Image.fromarray((self.current_image.binary_array * 255).astype(np.uint8))
        # Use Image.Resampling.LANCZOS for best downsampling quality
        binarized_photo = ImageTk.PhotoImage(binarized_image.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS))
        self.right_image_id = self.right_canvas.create_image(0, 0, anchor="nw", image=binarized_photo)
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

    def open_folder_selection_popup(self):
        # Hide the main window
        self.master.withdraw()

        # Create the popup window
        self.popup = Toplevel()
        self.popup.title("Folder Selection")

        # The centered label
        Label(self.popup, text="Please select the folders", anchor='center').pack(padx=10, pady=(10, 0))

        # The left-justified label with bullets
        Label(self.popup, text="1. Load spheroid grayscale images for binarizing\n"
                               "2. Save binarized images to", anchor='w', justify=LEFT).pack(padx=10, pady=(0, 10),
                                                                                             fill='x')

        Button(self.popup, text="Select Load Folder", command=self.select_load_folder).pack(padx=10, pady=5)
        Button(self.popup, text="Select Save Folder", command=self.select_save_folder).pack(padx=10, pady=5)

        self.continue_button = Button(self.popup, text="Continue", command=self.open_binarize_window, state=DISABLED)
        self.continue_button.pack(padx=10, pady=10)

        self.popup.protocol("WM_DELETE_WINDOW", self.on_close_popup)  # Handle the close event

    def open_folder_dialog(self, folder_type):
        folder_selected = filedialog.askdirectory()

        # Check that a folder has been selected
        if not folder_selected:
            return  # Skip the rest of the code if no folder is selected

        if folder_type == "load":
            self.image_folder_path = folder_selected
            self.image_list = [f for f in os.listdir(self.image_folder_path) if f.endswith('.tif')]
        elif folder_type == "save":
            self.save_folder_path = folder_selected

    def select_load_folder(self):
        self.open_folder_dialog("load")
        self.check_folder_selection()

    def select_save_folder(self):
        self.open_folder_dialog("save")
        self.check_folder_selection()

    def check_folder_selection(self):
        if self.image_folder_path and self.save_folder_path:
            self.continue_button['state'] = NORMAL

    def on_close_popup(self):
        """Close the popup window and reset related variables to their default values."""
        if self.popup:
            self.popup.destroy()  # Destroy the popup pane window
            self.popup = None  # Reset the window variable

        self.continue_button = None
        self.master.deiconify()

    def open_binarize_window(self):
        if self.popup:
            self.popup.destroy()  # Destroy the popup pane window
            self.popup = None  # Reset the window variablees
            self.continue_button = None

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
        self.zoom_in_button = None
        self.zoom_out_button = None
        self.image_frame = None

        # Reset drawing-related variables
        self.drawing = False
        self.delete_points()

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
        # Bring the modify pane to front if it's open
        if self.modify_pane and self.modify_pane.winfo_exists():
            self.modify_pane.lift()
            return

        # Create the modify pane as a Toplevel window
        self.modify_pane = Toplevel(self.binarize_window)
        self.modify_pane.title("Modifications")
        self.modify_pane.protocol("WM_DELETE_WINDOW", self.on_close_modify_pane)  # Handle the close event

        # Instructions Pane
        Label(self.modify_pane, text="Instructions:\n- Use sliders to adjust thresholds and blur.\n"
                                     "- Toggle boundary for automatic contour detection.\n"
                                     "- Apply to confirm changes.", justify=LEFT).pack(side='right', fill='y')

        # Local Threshold Slider
        Label(self.modify_pane, text="Set a threshold either globally or within a boundary").pack()
        self.local_thresh_scale = Scale(self.modify_pane, from_=0, to_=255, orient=HORIZONTAL,
                                     command=self.update_threshold, label="Threshold")
        self.local_thresh_scale.bind("<ButtonRelease-1>", self.on_threshold_slide_end)
        self.local_thresh_scale.set(int(self.recent_threshold))
        self.local_thresh_scale.pack(fill='x')

        # Blur Slider
        Label(self.modify_pane, text="Set a Gaussian blur for automatic boundary detection.").pack()
        self.blur_scale = Scale(self.modify_pane, from_=1, to_=11, resolution=2, orient=HORIZONTAL,
                                label="Blur", command=self.update_canvas)
        self.blur_scale.pack(fill='x')

        # Boundary Toggle Button
        Label(self.modify_pane, text="Toggle to draw automatic boundaries.").pack()
        self.boundary_var = IntVar()
        self.boundary_button = Button(self.modify_pane, text="Auto-Detect Boundary", command=self.toggle_boundary)
        self.boundary_button.pack()

        # Apply Button
        Label(self.modify_pane, text="Apply local threshold or boundary changes.").pack()
        self.apply_button = Button(self.modify_pane, text="Auto Clean", command=self.apply_modifications)
        self.apply_button.pack()

        # Draw Checkbo
        Label(self.modify_pane, text="Enable drawing mode for local boundary.").pack()
        self.draw_var = IntVar()
        self.draw_check = Checkbutton(self.modify_pane, text="Draw", variable=self.draw_var)
        self.draw_check.pack()

        # Add buttons to undo and redo changes
        self.undo_button = Button(self.modify_pane, text="Undo", command=self.undo_action)
        self.undo_button.pack()
        self.redo_button = Button(self.modify_pane, text="Redo", command=self.redo_action)
        self.redo_button.pack()

    def on_threshold_slide_end(self, *args):
        # Clear the points after applying the local threshold
        if len(self.points) <= 2:
            self.recent_threshold = self.local_threshold
        else:
            self.set_scale_loc_only = True
            self.local_thresh_scale.set(self.recent_threshold)

        self.delete_points()

        # Save the new image state
        self.save_state()

    def undo_action(self):
        if self.current_state_index > 0:
            self.restore_state(self.current_state_index - 1)

    def redo_action(self):
        if self.current_state_index < len(self.image_states) - 1:
            self.restore_state(self.current_state_index + 1)

    def toggle_boundary(self):
        self.boundary_var.set(not self.boundary_var.get())
        # Update the canvas with potential modifications
        self.update_canvas()

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
        self.threshold_scale = None
        self.set_scale_loc_only = False

    def apply_modifications(self):

        # Apply a local threshold of 256 to all contours except the largest
        if self.current_image.contours and self.boundary_var.get():
            mask = np.zeros_like(self.current_image.grayscale_array, dtype=np.uint8)
            for contour in self.current_image.contours[1:]:
                cv2.fillPoly(mask, [contour], 1)
            self.current_image.binary_array[mask.astype(bool)] = 0

        # Save the new image state
        self.save_state()

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
            
            
class SpheroidAnalysisApp:
    def __init__(self, root):
        self.root = root

        self.folder_label = None
        self.folder_button = None
        self.id_dict_frame = None
        self.progress = None
        self.run_button = None
        self.selected_folder = ""
        self.id_dict_entries = []
        self.id_dict_keys = ['experiment #']

    def open_analyze_window(self):
        # Hide the main window
        self.root.withdraw()

        # Create the binarize window
        self.analyze_window = Toplevel()
        self.analyze_window.title("Analyze Image")
        self.analyze_window.protocol("WM_DELETE_WINDOW", self.on_close_analyze_window)  # Handle the close event

        # Folder selection
        self.folder_label = Label(self.analyze_window, text="Select Folder:")
        self.folder_label.grid(row=0, column=0, sticky='w')
        self.folder_button = Button(self.analyze_window, text="Browse", command=self.select_folder)
        self.folder_button.grid(row=0, column=1)

        # ID dictionary table
        self.id_dict_frame = Frame(self.analyze_window)
        self.id_dict_frame.grid(row=1, column=0, columnspan=2)
        self.id_dict_entries = []
        self.create_id_dict_ui()

        # Progress bar
        self.progress = ttk.Progressbar(self.analyze_window, orient=HORIZONTAL, length=300, mode='determinate')
        self.progress.grid(row=2, column=0, columnspan=2, sticky='we')

        # Run button
        self.run_button = Button(self.analyze_window, text="Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=3, column=0, columnspan=2)


    def on_close_analyze_window(self):
        # Destroy and reset UI components to None
        self.build_id_dict()

        if self.analyze_window:
            self.analyze_window.destroy()  # Destroy the analysis window
            self.analyze_window = None  # Reset the window variable

        # Reset all components to None
        self.folder_label = None
        self.folder_label = None
        self.folder_button = None
        self.id_dict_frame = None
        self.progress = None
        self.run_button = None

        self.root.deiconify()

    def add_id_dict_row(self):
        row = Frame(self.id_dict_frame)
        key_entry = Entry(row)
        value_entry = Entry(row)
        key_entry.pack(side=LEFT)
        value_entry.pack(side=LEFT)
        row.pack(side=TOP, fill=X)
        self.id_dict_entries.append((key_entry, value_entry))

    def remove_id_dict_row(self):
        if len(self.id_dict_entries) > 1:
            key_entry, value_entry = self.id_dict_entries.pop()
            key_entry.destroy()
            value_entry.destroy()

            # Remove the last key from self.id_dict_keys
            if self.id_dict_keys:
                self.id_dict_keys.pop()

    def create_id_dict_ui(self):
        add_button = Button(self.id_dict_frame, text="+", command=self.add_id_dict_row)
        add_button.pack(side=LEFT)
        remove_button = Button(self.id_dict_frame, text="-", command=self.remove_id_dict_row)
        remove_button.pack(side=LEFT)

        for key in self.id_dict_keys:
            self.add_id_dict_row()
            self.id_dict_entries[-1][0].insert(0, key)  # Set the last key entry to the value of 'key'

    def build_id_dict(self):
        id_dict = {}
        keys = []

        for key_entry, value_entry in self.id_dict_entries:
            key = key_entry.get()
            value = value_entry.get()
            if key and value:
                id_dict[key] = value
                keys.append(key)  # Update self.id_dict_keys with the new keys
            elif key:
                id_dict[key] = ''
                keys.append(key)  # Update self.id_dict_keys with the new keys

        self.id_dict_keys = keys

        return id_dict

    def select_folder(self):
        self.selected_folder = filedialog.askdirectory()
        self.folder_label.config(text=f"Selected Folder: {self.selected_folder}")

    def run_analysis(self):
        # Start the analysis in a new thread
        analysis_thread = threading.Thread(target=self.analysis_logic)
        analysis_thread.start()

    def analysis_logic(self):
        data_fldr = self.selected_folder
        master_id_dict = self.build_id_dict()

        # Schedule progress bar update in the main thread
        progress_update = lambda p: self.analyze_window.after(0, self.update_progress_bar, p)

        analysis_logic(data_fldr, master_id_dict, progress_update)

        # Complete the progress bar
        self.analyze_window.after(0, self.update_progress_bar, 100)

    def update_progress_bar(self, value):
        self.progress['value'] = value


def main():
    # Create the main window (root of the Tk interface)
    root = Tk()
    # Set the dimensions of the window
    root.geometry("800x600")

    # Create the application
    app = MainMenu(root)

    # Start the application loop
    root.mainloop()


# Run the main application loop
if __name__ == '__main__':
    main()