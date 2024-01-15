# Importing required modules for GUI and image processing
from tkinter import Canvas, Toplevel, Checkbutton, IntVar, Tk, Frame, Button, Label, Entry, filedialog\
    , messagebox, Scale, HORIZONTAL, LEFT, TOP, X, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from constants import *
import pandas as pd
import re
import threading

from binarize import BinarizedImage
from quantify_image_set import QuantSpheroidSet, PlotPixelDistancesandAngles


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
        self.binarize_button = Button(self.frame, text="Binarize", command=self.binarize_ap.open_binarize_window)
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
        self.current_image.save_binarized_image()

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
            self.save_binarized_image()
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
        self.blur_scale = Scale(self.modify_pane, from_=1, to_=11, resolution=2, orient=HORIZONTAL, label="Blur")
        self.blur_scale.pack(fill='x')

        # Change Boundary Checkbox to Button
        self.boundary_var = IntVar()
        self.boundary_button = Button(self.modify_pane, text="Toggle Boundary", command=self.toggle_boundary)
        self.boundary_button.pack()

        # Apply Button
        self.apply_button = Button(self.modify_pane, text="Apply", command=self.apply_modifications)
        self.apply_button.pack()

        # Draw Checkbox
        self.draw_var = IntVar()
        self.draw_check = Checkbutton(self.modify_pane, text="Draw", variable=self.draw_var)
        self.draw_check.pack()

        # Update the state of the Local Threshold slider based on the points
        self.update_local_threshold_slider()

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

        # Apply a local threshold of 256 to all contours except the largest
        if self.current_image.contours and self.boundary_var.get():
            mask = np.zeros_like(self.current_image.grayscale_array, dtype=np.uint8)
            for contour in self.current_image.contours[1:]:
                cv2.fillPoly(mask, [contour], 1)
            self.current_image.binary_array[mask.astype(bool)] = 0

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
        if self.analyze_window:
            self.analyze_window.destroy()  # Destroy the binarize window
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

    def create_id_dict_ui(self):
        add_button = Button(self.id_dict_frame, text="+", command=self.add_id_dict_row)
        add_button.pack(side=LEFT)
        remove_button = Button(self.id_dict_frame, text="-", command=self.remove_id_dict_row)
        remove_button.pack(side=LEFT)
        self.add_id_dict_row()  # Add initial row

    def build_id_dict(self):
        id_dict = {}
        for key_entry, value_entry in self.id_dict_entries:
            key = key_entry.get()
            value = value_entry.get()
            if key and value:
                id_dict[key] = value
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
        id_dict = self.build_id_dict()
        print('Analysis started')

        # Filtering out directories from image_fpaths
        image_fpaths = []

        for f in os.listdir(data_fldr):
            _, img_ext = os.path.splitext(f)

            if img_ext in ['.tif', '.png', '.jpg']:
                image_fpaths.append(f)

        process_masked = True
        processed_experiments = []

        overall_summary_dataframe = pd.DataFrame()

        for i, fname in enumerate(image_fpaths):

            # Update progress bar
            print(f'Quantifying data {100 * i / len(image_fpaths):.1f}% complete')
            progress = 100 * i / len(image_fpaths)

            # Schedule progress bar update in the main thread
            self.analyze_window.after(0, self.update_progress_bar, progress)

            _, img_ext = os.path.splitext(fname)

            if not len(img_ext):
                continue

            _, ext = os.path.splitext(fname)
            spheroid_num = int(fname.split('_')[0])
            day = int(re.search(PATTERN, fname).group(1))
            is_masked = fname.split('_')[-1][:-len(ext)] == MASKED

            # Check if this experiment was already processed or has the propper masking
            if (spheroid_num in processed_experiments) or (process_masked != is_masked):
                continue

            fpaths_for_this_experiment = []

            for filename in image_fpaths:
                if (int(filename.split('_')[0]) == spheroid_num) and (filename.split('_')[-1][:-len(ext)] == MASKED):
                    fpaths_for_this_experiment.append(os.path.join(data_fldr, filename))

            # Make a folder to store the data from this experiment
            save_prefix = f'spheroid-{spheroid_num}'

            for id_name, id_value in id_dict.items():
                save_prefix = f'{id_name}-{id_value}_' + save_prefix

            save_fldr_path = os.path.join(data_fldr, save_prefix + '_data')

            if not os.path.isdir(save_fldr_path):
                os.makedirs(save_fldr_path)

            image_set_for_this_experiment = QuantSpheroidSet(fpaths_for_this_experiment)
            distances, indices, pixles, angles, outer_coordinates = image_set_for_this_experiment.distances_outside_initial_boundary()

            A0 = np.sum(image_set_for_this_experiment.images[0].img_array)

            areas = []
            Irb_values = []
            Ixb_values = []
            Iyb_values = []
            Irc_values = []
            Ixc_values = []
            Iyc_values = []
            max_speeds = []
            mean_speeds = []
            median_speeds = []
            max_angles = []
            mean_angles = []
            median_angles = []
            pa0_values = []
            pa1_values = []
            ps0_values = []
            ps1_values = []
            prin_speed_diff_values = []
            principle_Irb_values = []
            principle_Ixb_values = []
            principle_Iyb_values = []

            speed_values = np.zeros(angles.shape)
            speed_angle_columns_data = []

            for j in range(0, len(image_set_for_this_experiment.images) - 1):
                img, t = image_set_for_this_experiment.images[j + 1], image_set_for_this_experiment.times[j + 1]

                distances = distances[0]
                metrics = PlotPixelDistancesandAngles(save_fldr_path, t, distances, angles[j], outer_coordinates[0]
                                                      , np.sqrt(pixles[0, ::, 0] ** 2 + pixles[0, ::, 1] ** 2),
                                                      pixles[j], 2, 1)
                Irb, Ixb, Iyb, Irc, Ixc, Iyc, outerdistance_lengths, outer_distances_xy, centerdistance_lengths \
                    , full_distances_xy, speed_array, speed_dimensionalized = metrics

                areas.append(np.sum(img.img_array))
                Irb_values.append(Irb)
                Ixb_values.append(Ixb)
                Iyb_values.append(Iyb)
                Irc_values.append(Irc)
                Ixc_values.append(Ixc)
                Iyc_values.append(Iyc)
                max_speeds.append(np.max(speed_dimensionalized))
                mean_speeds.append(np.mean(speed_dimensionalized))
                median_speeds.append(np.median(speed_dimensionalized))
                max_angles.append(np.max(angles[j, :]))
                mean_angles.append(np.mean(angles[j, :]))
                median_angles.append(np.median(angles[j, :]))

                pca_metrics = img.pca(save_fldr_path, angles[j], speed_dimensionalized, t)
                pa0, pa1, ps0, ps1, prin_speed_difference, principle_Irb, principle_Ixb \
                    , principle_Iyb, transformed_angles, transformed_speeds = pca_metrics

                # Append to the columns list with appropriate names
                speed_angle_columns_data.append(('Speeds at time {}'.format(t), speed_dimensionalized))
                speed_angle_columns_data.append(('Angles at time {}'.format(t), angles[j, :]))
                speed_angle_columns_data.append(('PCA transformed Speeds at time {}'.format(t), transformed_speeds))
                speed_angle_columns_data.append(('PCA transformed Angles at time {}'.format(t), transformed_angles))

                pa0_values.append(pa0)
                pa1_values.append(pa1)
                ps0_values.append(ps0)
                ps1_values.append(ps1)
                prin_speed_diff_values.append(prin_speed_difference)
                principle_Irb_values.append(principle_Irb)
                principle_Ixb_values.append(principle_Ixb)
                principle_Iyb_values.append(principle_Iyb)

            # Create a dictionary of the summary values
            summary_dict = {id_name: [id_value] * (len(image_set_for_this_experiment.images) - 1) for id_name, id_value
                            in id_dict.items()}

            summary_dict.update({'t0 areas': A0 * np.ones(len(areas)),
                            'areas': areas,
                            'Irb': Irb_values,
                            'Ixb': Ixb_values,
                            'Iyb': Iyb_values,
                            'Irc': Irc_values,
                            'Ixc': Ixc_values,
                            'Iyc': Iyc_values,
                            'max_speed': max_speeds,
                            'mean_speed': mean_speeds,
                            'median_speed': median_speeds,
                            'max_angle': max_angles,
                            'mean_angle': mean_angles,
                            'median_angle': median_angles,
                            'principle0_angles': pa0_values,
                            'principle1_angles': pa1_values,
                            'principle0_speeds': ps0_values,
                            'principle1_speeds': ps1_values,
                            'prin_speed_difference': prin_speed_diff_values,
                            'principle_Irb': principle_Irb_values,
                            'principle_Ixb': principle_Ixb_values,
                            'principle_Iyb': principle_Iyb_values
                            })

            summary_dataframe = pd.DataFrame(summary_dict, index=image_set_for_this_experiment.times[1:])
            summary_dataframe.to_csv(os.path.join(save_fldr_path, 'summary.csv'))
            # Concatenate current summary dataframe to overall summary dataframe
            overall_summary_dataframe = pd.concat([overall_summary_dataframe, summary_dataframe])

            # Create a dictionary from the speed and angles column data
            data_dict = dict(speed_angle_columns_data)

            # Create the DataFrame from the dictionary
            speed_angle_dataframe = pd.DataFrame(data_dict)
            speed_angle_dataframe.to_csv(os.path.join(data_fldr, save_prefix + '_speeds_and_angles.csv'))

        # Save the overall summary dataframe to CSV at the end of the outermost loop
        # TODO add file names as a column in results table
        overall_summary_dataframe.to_csv(os.path.join(data_fldr, 'overall_summary.csv'))
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