
# SpheroidInvasionAnalysis

## Overview

This application consists of three main files: `quantify_image_set.py`, `binarize.py`, and `gui.py`. Each file has specific roles in the image analysis workflow, from binarizing images to analyzing and generating plots.

### quantify_image_set.py

This file contains the main logic for analyzing spheroid images. It defines several classes and functions:

- **SpheroidImage**: Represents a binarized spheroid image and provides methods to analyze the image, such as finding boundaries, performing PCA, and calculating distances.
- **QuantSpheroidSet**: Processes a set of images, sorts them by time, and loads them for analysis.
- **Various Functions**: Support functions for plotting and calculating metrics.

### binarize.py

This file handles the binarization of images, providing a class to load a grayscale image, apply a threshold to binarize it, and find contours.

- **BinarizedImage**: Handles the loading, binarization, and processing of grayscale images. It includes methods for updating masks, finding contours, calculating centroids, and saving the binarized images.

### gui.py

This file defines the graphical user interface (GUI) for the application, built using Tkinter. The GUI allows users to:

- **MainMenu**: The main menu that provides access to three functionalities: binarize images, analyze images, and consolidate CSV files.
- **ImageBinarizationApp**: Handles the binarization of images, including loading images, applying thresholds, drawing boundaries, and saving the results.
- **SpheroidAnalysisApp**: Manages the analysis of spheroid images, allowing users to select folders, input metadata, and run the analysis.
- **CSVConcatenatorApp**: Allows users to select and concatenate multiple CSV files into a single file.

## Detailed ReadMe and Usage Guide

### Installation

When running on Windows, the application comes with everything needed to run it with embedded Python.
When installing on Linux or mac ensure you have the necessary dependencies installed. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Usage Guide

#### Main Menu

When you run the application, the main window opens, providing access to three main functions:

- **Binarize**: Opens the Image Binarization App.
- **Analyze**: Opens the Spheroid Analysis App.
- **Consolidate**: Opens the CSV Concatenator App.

#### Binarize Images

1. **Open the Binarize Window**: Click on "Binarize" from the main menu.
2. **Select Folders**:
    - Click "Select Load Folder" to choose the folder containing the images you want to binarize.
    - Click "Select Save Folder" to choose the folder where the binarized images will be saved.
3. **Binarize Window**:
    - The images will be displayed side-by-side: the original grayscale image on the left and the binarized image on the right.
    - Use the "Threshold" slider to adjust the binarization threshold.
    - Draw boundaries on the binarized image to apply local thresholds or delete regions.
    - Use the "Blur" slider to apply Gaussian blurring for better contour detection.
    - Click "Auto-Detect Boundary" to automatically detect and draw contours.
    - Use the "Save" button to save the binarized images.
4. **Settings Page**:
    - Access the settings page to adjust time unit, pixel scale, font, and tick sizes.

#### Analyze Images

1. **Open the Analyze Window**: Click on "Analyze" from the main menu.
2. **Select Folder**: Click "Browse" to choose the folder containing the images to be analyzed.
3. **Enter Metadata**: Provide optional metadata for the analysis, such as experiment number or condition.
4. **Run Analysis**: Click "Run Analysis" to start processing the images. Progress will be shown in a new window.

#### Consolidate CSV Files

1. **Open the Consolidate Window**: Click on "Consolidate" from the main menu.
2. **Select CSV Files**: Click "Select CSV Files" to choose the CSV files you want to concatenate.
3. **Remove Selected Files**: Use the "Remove Files" button to remove any selected files from the list.
4. **Concatenate Files**: Click "Concatenate Files" to merge the selected CSV files into one. You will be prompted to choose a location to save the concatenated file.

### Running the Application

To run the application, execute the `gui.py` file:

```bash
python gui.py
```

Or double click `run.bat` file on Windows

## Usage Guide for Direct Code Usage

If you prefer to use the code directly without the GUI, follow these instructions for binarizing images, analyzing spheroid data, and consolidating CSV files.

### Setup

Ensure you have the necessary dependencies installed. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Binarize Images

To binarize images directly using the `BinarizedImage` class in `binarize.py`:

1. **Import the Class**:

    ```python
    from binarize import BinarizedImage
    ```

2. **Load and Binarize an Image**:

    ```python
    # Example path to an image file
    image_path = 'path/to/image.tif'
    save_path = 'path/to/save_directory'

    # Initialize the BinarizedImage object
    bin_image = BinarizedImage(image_path, save_path, threshold=36)

    # Apply a different threshold if needed
    new_threshold = 50
    bin_image.update_mask(new_threshold)

    # Optionally, apply Gaussian blur and find contours
    bin_image.auto_contour(guassian_kernel=(5, 5))

    # Save the binarized image
    bin_image.save_binarized_image()
    ```

3. **Working with Contours and Centroids**:

    ```python
    # Find the centroid of the spheroid
    centroid = bin_image.find_spheroid_centroid()

    # Create and apply a circular mask centered at the centroid
    bin_image.create_circular_mask()
    ```

### Analyze Spheroid Images

To analyze a set of spheroid images using the `QuantSpheroidSet` class in `quantify_image_set.py`:

1. **Import the Necessary Classes and Functions**:

    ```python
    from quant import QuantSpheroidSet, analysis_logic
    from queue import Queue
    ```

2. **Set Up Analysis Parameters**:

    ```python
    data_fldr = 'path/to/spheroid_images'
    save_path = 'path/to/save_directory'
    pattern = r'day(\d+)'  # Example pattern to extract time points from filenames
    time_unit = 'day'
    pixel_scale = 1.0  # Microns per pixel
    font_spec = {'fontname': 'Arial', 'size': 12}
    tick_size = 11

    # Initialize the kill queue
    kill_queue = Queue()

    # Metadata dictionary
    master_id_dict = {'experiment #': 19, 'condition': 'dynamic'}

    # Run the analysis
    summary_path = analysis_logic(data_fldr, master_id_dict, print, kill_queue, pattern, time_unit, pixel_scale, font_spec, tick_size, save_images_to_pdf=False)

    print(f'Analysis complete. Summary saved at: {summary_path}')
    ```
