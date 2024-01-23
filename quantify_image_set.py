import tkinter as tk
import threading
from tkinter import filedialog, ttk
import os
from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from sklearn.decomposition import PCA
from constants import *


ARIAL = {'fontname': 'Arial',
         'size'    : 22}



class SpheroidImage:
    """
    Class representing a binarized spheroid image.

    Args:
        fpath (str): The file path to the spheroid image.

    Attributes:
        boundary (array): The boundary coordinates of the largest spheroid in the image
        centroid (array): The x y coordinates for the center of the boundary
        img_array (array): The spheroid image stored in array form
        x_coords (array): The x coordinates of every pixel in img_array
        y_coords (array): The y coordinates of every pixel in img_array
    """

    def __init__(self, fpath):
        source_image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

        # Find contours in the source image
        contours, _ = cv2.findContours(source_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour based on area
        largest_contour = np.array(max(contours, key=cv2.contourArea))

        # Calculate moments of the binary image for centering the contour
        M = cv2.moments(largest_contour)
        X = int(M["m10"] / M["m00"])
        Y = int(M["m01"] / M["m00"])

        self.boundary = largest_contour
        self.centroid = np.array([X, Y])
        self.img_array = source_image
        self.fname = Path(fpath).stem

        # Generate a range of numbers for the width and height
        x_range = np.arange(source_image.shape[0])
        y_range = np.arange(source_image.shape[1])

        # Create meshgrid of coordinates
        self.x_coords, self.y_coords = np.meshgrid(y_range, x_range)

    def center_boundary(self, boundary):
        """
        Centers the input boundary with the spheroid centroid

        Args:
            boundary (array-like): The coordinates defining the boundary of the spheroid object.

        Returns:
            array: The centered boundary
        """

        # Calculate moments of the binary image for centering the contour
        M = cv2.moments(boundary)
        bX = int(M["m10"] / M["m00"])
        bY = int(M["m01"] / M["m00"])

        # Translate the contour to the center of the target image
        return boundary + self.centroid - np.array([bX, bY])

    def get_pixle_coor_outside_boundary(self, boundary):
        """
        Find the coordinates for non zero pixels outside the boundary

        Args:
            boundary (array-like): The coordinates defining the boundary of the spheroid object.

        Returns:
            array: The centered boundary
        """

        # Create a mask for the region inside the boundary
        mask = np.ones_like(self.img_array, dtype=np.uint8)

        # Filter the image pixles to exclude the ones inside the boundary
        cv2.fillPoly(mask, [np.array(boundary)], False)
        mask = mask.astype(bool)

        # Filte the x and y coordinates to only contain the ones that are non zero and outside the bondary
        x_coor_outside_bound = self.x_coords[mask][self.img_array[mask] > 0]
        y_coor_outside_bound = self.y_coords[mask][self.img_array[mask] > 0]

        return np.stack((x_coor_outside_bound, y_coor_outside_bound), axis=1)

    def intersection_distance(self, boundary):
        """
        intersection_distance finds the minimum distance from the initial boundary to the pixels outside the boundary.
        measured as the length of the line between the spheroid centroid (centroid_loc, x0) and the invaded cell pixels
        (outer_pixels, x1). The function zeros the coordinate system at x0.

        Args:
            boundary (array): The coordinates defining the boundary of the spheroid object.

        Returns:
            array: The distances from the outer pixels to the boundary as measured along a straight line drawn from
                the spheroid center (centroid) to each pixel, not the shortest distance to the edge.

            array: The index of the boundary pixels that intersect the line from the spheroid centroid to the
                outerpixels. This array is the same shape os the outer pixels array and contains a boundary index for
                each outer pixel

            array: The outer pixel coordinates centered on (0, 0)

            array: The boundary coordinates centered on (0, 0)

        """

        outer_pixels_full = self.get_pixle_coor_outside_boundary(boundary)

        # Setup batching to avoid out of memory errors on smaller CPUs
        batch_size = 10000
        num_pix = len(outer_pixels_full)
        num_batches = num_pix // batch_size + (num_pix % batch_size != 0)

        # Setup initial arrays to fill with data each batch
        distance_magnitude = np.zeros(num_pix)
        close_inds = np.zeros(num_pix)
        boundary_pixels_full = np.zeros(outer_pixels_full.shape)

        for i in range(num_batches):
            # Select the outer pixels for this batch
            outer_pixels = outer_pixels_full[i * batch_size : min((i + 1) * batch_size, num_pix)]

            # Define coordinates and center everything along the centroid location to make calculations
            # easier
            x0 = self.centroid[0]
            y0 = self.centroid[1]
            x1 = outer_pixels[:, 0].T - x0
            y1 = outer_pixels[:, 1].T - y0  # a (1 x N) array
            xb = boundary[:, 0] - x0
            yb = boundary[:, 1] - y0  # a (M x 1) array

            # Reshape to utilize vector operations
            N = len(x1)  # number of pixels
            M = len(yb)  # number of boundary points

            xb_reshaped = np.tile(xb, (N, 1)).T  # shape M, N = (MxN) array
            yb_reshaped = np.tile(yb, (N, 1)).T  # shape M, N
            x1_reshaped = np.tile(x1, (M, 1))  # shape M, N
            y1_reshaped = np.tile(y1, (M, 1))  # shape M, N

            # Define slopes
            m = y1_reshaped / (x1_reshaped + 1e-8) # shape M, N

            # Calculate the minimum distance from each boundary point to each line
            x_min = (xb_reshaped + m * yb_reshaped) / (1 + m ** 2)  # shape M, N
            y_min = m * x_min  # shape M, N
            d = np.sqrt((xb_reshaped - x_min) ** 2 + (yb_reshaped - y_min) ** 2)  # shape M, N

            # Filter out the parts of the boundary that are unwanted --------------- #

            # - Part 1: The half of the boundary that is on the opposite side from the pixel of interest (POI)
            #  - not nearest the POI
            perp_x1 = 1
            perp_x2 = -1
            perp_y1 = (-1 / (m + 1e-8))
            perp_y2 = (1 / (m + 1e-8))
            perp_valb = (xb_reshaped - perp_x1) * (perp_y2 - perp_y1) - (yb_reshaped - perp_y1) * (perp_x2 - perp_x1)
            perp_val_locs = (x1_reshaped - perp_x1) * (perp_y2 - perp_y1) - (y1_reshaped - perp_y1) * (perp_x2 - perp_x1)
            perp_mask = np.not_equal((perp_valb / np.abs(perp_valb)),  (perp_val_locs / np.abs(perp_val_locs)))

            # - Part 2: Filters out parts of boundary that are overlapping/ further from
            # - the centroid than the POI
            dist_mask = (xb_reshaped ** 2 + yb_reshaped ** 2) > (x1_reshaped ** 2 + y1_reshaped** 2)

            # - Part 3: Apply filters
            tot_mask = np.logical_or(perp_mask, dist_mask)
            d[tot_mask] = np.max(d)

            # -------------------------------------------------------------------- #

            # Find the intercept by finding the boundary point closest to the line
            current_close_inds = np.argmin(d, axis=0) # shape will be (N,)
            close_inds[i * batch_size: min((i + 1) * batch_size, num_pix)] = current_close_inds

            # calculate the distance magnitude
            xb_close = xb[current_close_inds] # N
            yb_close = yb[current_close_inds] # N
            distance_magnitude[i * batch_size : min((i + 1) * batch_size, num_pix)] = np.sqrt((x1 - xb_close) ** 2 + (y1 - yb_close) ** 2)  # shape N
            boundary_pixels_full[i * batch_size : min((i + 1) * batch_size, num_pix)] = np.stack((xb_close, yb_close), axis=1)

        return distance_magnitude, close_inds, outer_pixels_full - self.centroid, boundary_pixels_full

    def pca(self, save_fldr_path, angles, speeds, t):
        """
        Perform Principal Component Analysis (PCA) on velocity data and generate plots.
        This function takes velocity data in terms of angles and speeds, applies PCA to transform the data,
        and then plots the original and transformed data. It also calculates some statistics - principal
        angles, speeds, and moments of inertia.

        Args:
            save_fldr_path (str): The file path where the plots will be saved.
            angles (array): The angular components of the velocity data.
            speeds (array): The speed components of the velocity data.
            t (int/float): A time parameter used in the calculation of distances and moments.

        Returns:
            tuple: Contains principal angles, speeds, speed difference, and moments of inertia.
        """

        # Convert polar coordinates (angle, speed) to Cartesian coordinates (x, y)
        velocities = speeds[::, np.newaxis] * np.stack((np.cos(angles), np.sin(angles)), axis=1)

        # Initialize PCA and fit it to the velocities
        pca = PCA()
        score = pca.fit_transform(velocities)
        coeff = pca.components_
        latent = pca.explained_variance_

        # Transform the angles and speeds using the PCA scores
        transformed_angles = np.arctan2(score[:, 1], score[:, 0])
        transformed_speeds = np.sqrt(score[:, 1] ** 2 + score[:, 0] ** 2)

        # Define a function to plot an axis line at a given angle
        def plot_axis(ax, angle, max_radius):
            angle_rad = np.deg2rad(angle)  # Convert the angle to radians
            ax.plot([angle_rad, angle_rad], [0, max_radius], 'b')  # Plot in the positive direction
            ax.plot([angle_rad, angle_rad], [0, -max_radius], 'b')  # Plot in the negative direction as well

        # Calculate principal angles and speeds
        prin_angle1 = np.degrees(np.arctan2(coeff[1, 0], coeff[0, 0])) % 360
        prin_angle2 = np.degrees(np.arctan2(coeff[1, 1], coeff[0, 1])) % 360
        prin_speed1 = np.mean(np.abs(score[:, 0]))
        prin_speed2 = np.mean(np.abs(score[:, 1]))

        principle_angles = np.stack((prin_angle1, prin_angle2))
        principle_speeds = np.stack((prin_speed1, prin_speed2))

        # Plot original data
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(angles, speeds, '.')

        ax.set_title('Persistence speed (um/min) vs migration angle', **ARIAL)

        # Annotate plot with PCA information
        principle_angles_str = ', '.join(map(str, principle_angles))
        principle_speeds_str = ', '.join(map(str, principle_speeds))
        ann_str = [f'Principle angles: ({principle_angles_str})',
                   f'Mean principle speeds: ({principle_speeds_str})']
        ax.annotate('\n'.join(ann_str), xy=(0.5, -0.1), xycoords='axes fraction', fontsize=14)

        # Plot principal angles
        # Creating arrays for plotting lines representing principal angles
        a1, s1 = np.deg2rad([principle_angles[0], principle_angles[0]]), [0, max(speeds)]
        a2, s2 = np.deg2rad([principle_angles[0] + 180, principle_angles[0] + 180]), [0, max(speeds)]
        a3, s3 = np.deg2rad([principle_angles[1], principle_angles[1]]), [0, max(speeds)]
        a4, s4 = np.deg2rad([principle_angles[1] + 180, principle_angles[1] + 180]), [0, max(speeds)]
        ax.plot(a1, s1, "b")
        ax.plot(a2, s2, "b")
        ax.plot(a3, s3, "k")
        ax.plot(a4, s4, "k")
        ax.set_rlim(rmin=0)
        plt.savefig(os.path.join(save_fldr_path, f't{t}_speed_vs_angle_with_principle_components.png'))
        plt.close()

        # Plot transformed data
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(transformed_angles, transformed_speeds, '.')
        ax.set_title('Transformed persistence speed (um/min) vs migration angle', **ARIAL)

        # Plot principal axes on the transformed data plot
        a3, s3 = np.deg2rad([0, 0]), [0, max(speeds)]

        a4, s4 = np.deg2rad([180, 180]), [0, max(speeds)]
        a5, s5 = np.deg2rad([90, 90]), [0, max(speeds)]
        a6, s6 = np.deg2rad([-90, -90]), [0, max(speeds)]

        # Plotting four axes at 0, 90, 180, and -90 degrees
        ax.plot(a3, s3, "b")
        ax.plot(a4, s4, "b")
        ax.plot(a5, s5, "k")
        ax.plot(a6, s6, "k")
        plt.savefig(os.path.join(save_fldr_path, f't{t}_principle_coordinates_speed_vs_angle.png'))
        plt.close()

        # Calculate the difference between principal speeds
        prin_speed_difference = principle_speeds[0] - principle_speeds[1]

        # Calculate distances from boundary in pixels
        # Multiplying by time factor (t * 24 * 60) to get the distance
        transformed_outerdistance_lengths = transformed_speeds * (t * 24 * 60)# / pixel_size
        transformed_outerdistances_xy = score * (t * 24 * 60)# / pixel_size
    
        # Calculate moments of inertia from boundary
        # Assuming area for calculation = 1 for simplicity
        a = 1
        principle_Irb = np.sum(transformed_outerdistance_lengths ** 2 * a)
        principle_Ixb = np.sum(transformed_outerdistances_xy[:, 1] ** 2 * a)
        principle_Iyb = np.sum(transformed_outerdistances_xy[:, 0] ** 2 * a)

        # Return the calculated principal angles, speeds, speed difference, and moments of inertia
        return principle_angles[0], principle_angles[1], principle_speeds[0], principle_speeds[
            1], prin_speed_difference, principle_Irb, principle_Ixb, principle_Iyb, transformed_angles, transformed_speeds



    def get_angles_outside_boundary(self, boundary):
        """
        Calculate the angles of outer pixels relative to the spheroid centroid.

        Args:
            boundary (array): An array representing the coordinates for the boundary.

        Returns:
            array: An array of angles in radians for each outer pixel relative to the centroid.
        """

        # Get the coordinates of pixels outside the specified boundary
        outer_pixels = self.get_pixle_coor_outside_boundary(boundary)

        # Center the x-coordinates of the outer pixels relative to the centroid's x-coordinate
        # Prevent division by zero by ensuring that x-coordinates are not exactly zero
        centered_x_coor = outer_pixels[:, 0] - self.centroid[0]
        centered_x_coor = np.sign(centered_x_coor) * np.clip(np.abs(centered_x_coor), 1e-8, np.inf)

        # Center the y-coordinates of the outer pixels relative to the centroid's y-coordinate
        centered_y_coor = outer_pixels[:, 1] - self.centroid[1]

        # Calculate and return the angles in radians using arctan2 for each outer pixel
        return np.arctan2(centered_y_coor, centered_x_coor)


class QuantSpheroidSet:
    """
    This class processes a set of image file paths, sorts them based on time extracted from
    their filenames, and loads the images. It also sets the save folder path for any outputs.

    Args:
        image_fpaths (list): A list of file paths for the images.
        save_path (str, optional): The path where outputs should be saved. If None, uses the
                                   directory of the first image.

    Attributes:
        times (array): Sorted times extracted from image file names.
        paths (array): Image file paths sorted according to their corresponding times.
        images (list): A list of SpheroidImage objects loaded from the paths.
        save_fldr_path (str): Path to the folder where outputs will be saved.
    """

    def __init__(self, image_fpaths, save_path=None):
        # Extracting time information from the image filenames
        sample_times = np.array([int(re.search(PATTERN, os.path.basename(filename)).group(1))
                        for filename in image_fpaths if re.search(PATTERN, filename)])
        array_paths = np.array(image_fpaths)

        # Sorting the times and paths based on the extracted times
        self.times = np.sort(sample_times)
        self.paths = array_paths[sample_times.argsort()]

        # Loading images as SpheroidImage objects
        self.images = [SpheroidImage(fpath) for fpath in self.paths]

        # If no explicit save path save in the same directory as the images
        if save_path is None:
            self.save_fldr_path = os.path.dirname(image_fpaths[0])
        else:
            self.save_fldr_path = save_path

    def distances_outside_initial_boundary(self):
        """
        Calculate distances to the pixels outside the initial boundary for each image in the
        set, excluding the first image.

        Returns:
            tuple: arrays of distances, indices, coordinates, angles, and outer coordinates.
        """
        # Initial boundary is taken from the first image
        init_bound = self.images[0].boundary.squeeze()

        # Initializing lists to store calculated values
        distances = []
        angles_ls = []
        indices = []
        coordinates = []
        outer_coordinates = []
        times = []

        # Iterating over images (excluding the first) to calculate metrics
        for img in self.images[1:]:
            centered_boundary = img.center_boundary(init_bound)
            dist, inds, coor, boundary_coor = img.intersection_distance(centered_boundary)
            angles = img.get_angles_outside_boundary(centered_boundary)

            # Uncomment to test find distance to boundary
            # # Set up the figure and axis for the image and the slider
            # fig, ax = plt.subplots()
            # plt.subplots_adjust(bottom=0.25)  # Adjust subplot to make room for the slider
            #
            # # Initial plot setup
            # ax.imshow(img.img_array)
            # x_coords = centered_boundary[:, 0]
            # y_coords = centered_boundary[:, 1]
            # line, = ax.plot(x_coords, y_coords, 'r-', linewidth=2)  # Initial boundary line
            # point_bx, = ax.plot(x_coords[0], y_coords[0], 'bx')  # Initial blue x
            # point_rx, = ax.plot(coor[0, 0] + img.centroid[0], coor[0, 1] + img.centroid[1], 'r.')  # Initial red x
            #
            # # Slider setup
            # ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])  # Slider position and size
            # slider = Slider(ax_slider, 'Index', 0, len(x_coords) - 1, valinit=0, valfmt='%0.0f')
            #
            # # Update function to redraw the plot when the slider is changed
            # def update(val):
            #     ind = int(slider.val)
            #
            #     current_pixles = coor[inds == ind]
            #
            #     point_bx.set_data(x_coords[ind], y_coords[ind])
            #     point_rx.set_data(current_pixles[::, 0] + img.centroid[0], current_pixles[::, 1] + img.centroid[1])
            #     fig.canvas.draw_idle()
            #
            # # Call update function when slider value is changed
            # slider.on_changed(update)
            #
            # # Display the interactive plot
            # plt.show()

            # Uncomment to test find angles to boundary
            # def update(val):
            #     # Retrieve the current index from the slider
            #     ind = int(slider.val)
            #
            #     # Update the angle and point based on the new index
            #     angle_rad = angles[ind]
            #     pix_pt = coor[ind]
            #
            #     # Calculate the new end point of the line
            #     end_x = center[0] + length * np.cos(angle_rad)
            #     end_y = center[1] + length * np.sin(angle_rad)
            #
            #     # Calculate the new end point of the line
            #     start_x = center[0]# - length * np.cos(angle_rad)
            #     start_y = center[1]# - length * np.sin(angle_rad)
            #
            #     # Update the line and point on the plot
            #     line.set_xdata([start_x, end_x])
            #     line.set_ydata([start_y, end_y])
            #     point_rx.set_data(pix_pt[0] + center[0], pix_pt[1] + center[1])
            #
            #     # Redraw the figure
            #     fig.canvas.draw_idle()
            #
            # # Initial setup (this part is mostly from your original code)
            # center = img.centroid
            # length = 5000
            # ind = 10000  # This will be replaced by the slider value
            #
            # # Set up the figure and axis for the image and the slider
            # fig, ax = plt.subplots()
            # plt.subplots_adjust(bottom=0.25)  # Adjust subplot to make room for the slider
            #
            # # Initial plot setup
            # ax.imshow(img.img_array)
            # x_coords = centered_boundary[:, 0]
            # y_coords = centered_boundary[:, 1]
            # line, = ax.plot([], [], 'r-')  # Initial boundary line (empty for now)
            # xlim = plt.xlim()
            # ylim = plt.ylim()
            # point_rx, = ax.plot([], [], 'bx')  # Initial red x (empty for now)
            #
            # # Add a grid for better visualization
            # plt.grid(True)
            #
            # # Create the slider
            # ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])  # Position the slider
            # slider = Slider(ax_slider, 'Index', 0, len(coor) - 1, valinit=ind, valstep=1)
            #
            # # Call the update function when the slider value is changed
            # slider.on_changed(update)
            #
            # # Initialize the plot with the initial index
            # update(ind)
            #
            # # Show the plot with the slider
            # plt.show()

            distances.append(dist)
            indices.append(inds)
            coordinates.append(coor)
            outer_coordinates.append(coor - boundary_coor)
            angles_ls.append(angles)
            times.append(img)

        # Convert lists to numpy arrays
        distances = np.asarray(distances)
        indices = np.asarray(indices)
        coordinates = np.asarray(coordinates)
        angles_ls = np.asarray(angles_ls)
        outer_coordinates = np.asarray(outer_coordinates)

        return distances, indices, coordinates, angles_ls, outer_coordinates


def PlotPixelDistancesandAngles(save_fldr_path, t, outerdistance_lengths, angles_array, outer_distances_xy, centerdistance_lengths,
                                    full_distances_xy, num_days, pixel_size):
    """
    Plots distance histograms, cumulative distance histograms, angles histograms, representative distance
    values, and speed vs angle graphs. It also calculates and plots the moment of inertia from the boundary and center
    for the given pixel distances

    Args:
        save_fldr_path (str): The file path where the plots will be saved.
        t (int): Time point or identifier for the data set.
        outerdistance_lengths (array): Array of distances of each pixel from the boundary.
        angles_array (array): Array of angles corresponding to each pixel's position relative to a reference.
        outer_distances_xy (array): XY coordinates of pixels relative to the boundary.
        centerdistance_lengths (array): Array of distances of each pixel from the center.
        full_distances_xy (array): XY coordinates of pixels relative to the center.
        num_days (int): Number of days over which the data is collected.
        pixel_size (float): The size of a pixel in meters.

    Returns:
        tuple: A tuple containing moments of inertia from the boundary (Irb, Ixb, Iyb),
               moments of inertia from the center (Irc, Ixc, Iyc), the arrays of outerdistance lengths, outer distances xy,
               centerdistance lengths, full distances xy, speed array, and speed array converted to distance units.
    """

    plt.figure()

    # Plot distances in a histogram
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    h1, _, _ = plt.hist(outerdistance_lengths, bins=20)  # Histogram with 20 bins
    plt.title('Distances from boundary histogram', **ARIAL)
    plt.xlabel('distance (pixels)', **ARIAL)
    plt.ylabel('frequency', **ARIAL)

    # Plot the cumulative values in a histogram
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
    x = np.arange(50, 1001, 50)  # Values from 50 to 1000 with a step of 50
    cumDistValues = np.cumsum(h1)  # Cumulative sum of the histogram values
    plt.bar(x[:len(cumDistValues)], cumDistValues, width=45)  # Bar plot
    plt.title('Cumulative distances from boundary histogram')
    plt.xlabel('distance (pixels)', **ARIAL)
    plt.ylabel('frequency', **ARIAL)

    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.savefig(os.path.join(save_fldr_path, f't{t}_distances_histogram.png'))  # Save the figure
    plt.close()

    # Assuming 'angles' is a list of numpy arrays or lists
    #angles_array = np.concatenate(angles)  # Concatenate all angle arrays

    # Create a polar histogram
    plt.figure()
    ax = plt.subplot(111, polar=True)  # Create a polar subplot
    ax.hist(angles_array, bins=20)  # Polar histogram with 20 bins
    plt.title('Angles histogram', **ARIAL)
    plt.savefig(os.path.join(save_fldr_path, f't{t}_angles_histogram.png'))
    plt.close()

    max_dist = np.max(outerdistance_lengths)
    median_dist = np.median(outerdistance_lengths)
    mean_dist = np.mean(outerdistance_lengths)

    # Plotting the bar graph
    categories = ['max', 'median', 'mean']
    values = [max_dist, median_dist, mean_dist]

    plt.figure()
    bars = plt.bar(categories, values)

    plt.title('Representative distance values from the boundary')
    plt.ylabel('distance (pixels)')

    # Adding values on top of the bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(np.ceil(value))}',
                 ha='center', va='bottom')

    plt.savefig(os.path.join(save_fldr_path, f't{t}_representative_distances.png'))
    plt.close()

    plt.figure()
    ax1 = plt.subplot(111, polar=True)
    ax1.plot(angles_array, centerdistance_lengths, '.')
    plt.title("Distances from center (pixels) vs angle")
    plt.savefig(os.path.join(save_fldr_path, f't{t}_distances_vs_angle_center.png'))
    plt.close()

    # Plot the boundary distance vs angle values in a polar plot
    plt.figure()
    ax2 = plt.subplot(111, polar=True)
    ax2.plot(angles_array, outerdistance_lengths, '.')
    plt.title("Distances from boundary (pixels) vs angle")
    plt.savefig(os.path.join(save_fldr_path, f't{t}_distances_vs_angle_boundary.png'))
    plt.close()

    # Convert distances to meters and calculate speed
    outerdistance_lengths_m = outerdistance_lengths * pixel_size
    centerdistance_lengths_m = centerdistance_lengths * pixel_size
    speed_array = outerdistance_lengths_m / (num_days * 24 * 60)  # m/min

    # Plotting the speed vs angle
    plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.plot(angles_array, speed_array, '.')
    plt.title('Speed vs angle')
    plt.savefig(os.path.join(save_fldr_path, f't{t}_speed_vs_angle.png'))
    plt.close()

    a = 1  # area for calculation - in this case = 1 bc pixels

    # Calculate the area moment of inertia from the boundary
    Irb = np.sum(outerdistance_lengths.astype(np.float64) ** 2 * a)
    Ixb = np.sum(outer_distances_xy[:, 1].astype(np.float64) ** 2 * a)
    Iyb = np.sum(outer_distances_xy[:, 0].astype(np.float64) ** 2 * a)

    # Calculate the area moment of inertia from the spheroid center
    Irc = np.sum(centerdistance_lengths.astype(np.float64) ** 2 * a)
    Ixc = np.sum(full_distances_xy[:, 1].astype(np.float64) ** 2 * a)
    Iyc = np.sum(full_distances_xy[:, 0].astype(np.float64) ** 2 * a)

    def plot_moment_of_inertia(title, moments):
        categories = ['Ir', 'Ix', 'Iy']
        plt.figure()
        bars = plt.bar(categories, moments)
        plt.title(title)
        plt.ylabel('Moment (pixels^4)')
        plt.gca().tick_params(labelsize=14)

        # Adding values on top of the bars
        for bar, value in zip(bars, moments):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2e}',
                     ha='center', va='bottom')

        plt.savefig(os.path.join(save_fldr_path, f't{t}_' + title.replace(' ', '_').lower() + '.png'))
        plt.close()

    # Plot the moment of inertia from center
    plot_moment_of_inertia('Moment of inertia from center', [Irc, Ixc, Iyc])

    # Plot the moment of inertia from boundary
    plot_moment_of_inertia('Moment of inertia from boundary', [Irb, Ixb, Iyb])

    return Irb, Ixb, Iyb, Irc, Ixc, Iyc, outerdistance_lengths, outer_distances_xy, centerdistance_lengths, full_distances_xy, speed_array, pixel_size * speed_array


def quantify_progress_print(progress):
        print(f'Quantifying data {progress}% complete')


def analysis_logic(data_fldr, master_id_dict, progress_print_fun):
    """
    Loops through spheroid images and saves the relevant data for further analysis. Groups spheroids by their prefix
    number and characterizes them based on the time points in the file name expressed as <time unit>T.

    Args:
        data_fldr (str): The file path where the images are stored and the data will be saved
        master_id_dict (dict): Dictionary containing meta data for this set of spheroid images
        progress_print_fun (callable): A function to display the analysis progress
    """

    print('Analysis started')

    # Filtering out directories from image_fpaths
    image_fpaths = []

    for f in os.listdir(data_fldr):
        _, img_ext = os.path.splitext(f)

        if img_ext in ['.tif', '.png', '.jpg']:
            image_fpaths.append(f)

    processed_experiments = []

    overall_summary_dataframe = pd.DataFrame()

    for i, fname in enumerate(image_fpaths):

        # Update progress bar
        progress = 100 * i / len(image_fpaths)
        progress_print_fun(progress)

        _, img_ext = os.path.splitext(fname)

        if not len(img_ext):
            continue

        _, ext = os.path.splitext(fname)
        spheroid_num = int(fname.split('_')[0])

        # Check if this experiment was already processed or has the propper masking
        if (spheroid_num in processed_experiments):
            continue

        processed_experiments.append(spheroid_num)
        fpaths_for_this_experiment = []

        for filename in image_fpaths:
            if (int(filename.split('_')[0]) == spheroid_num):
                fpaths_for_this_experiment.append(os.path.join(data_fldr, filename))

        # Make a folder to store the data from this experiment
        id_dict = {'spheroid': spheroid_num}
        id_dict.update(master_id_dict.copy())
        save_prefix = ''

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
        fname_list = []
        speed_angle_columns_data = []

        for j in range(0, len(image_set_for_this_experiment.images) - 1):
            img, t = image_set_for_this_experiment.images[j + 1], image_set_for_this_experiment.times[j + 1]

            distances = distances[j]
            metrics = PlotPixelDistancesandAngles(save_fldr_path, t, distances, angles[j], outer_coordinates[j]
                                                  , np.sqrt(pixles[j, ::, 0] ** 2 + pixles[j, ::, 1] ** 2),
                                                  pixles[j], 2, 1)
            Irb, Ixb, Iyb, Irc, Ixc, Iyc, outerdistance_lengths, outer_distances_xy, centerdistance_lengths \
                , full_distances_xy, speed_array, speed_dimensionalized = metrics

            fname_list.append(img.fname)
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

        summary_dict.update({
            'file': fname_list,
            'times': image_set_for_this_experiment.times[1:],
            't0 areas': A0 * np.ones(len(areas)),
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

        summary_dataframe = pd.DataFrame(summary_dict)
        summary_dataframe.to_csv(os.path.join(save_fldr_path, 'summary.csv'), index=False)
        # Concatenate current summary dataframe to overall summary dataframe
        overall_summary_dataframe = pd.concat([overall_summary_dataframe, summary_dataframe])

        # Create a dictionary from the speed and angles column data
        data_dict = dict(speed_angle_columns_data)

        # Create the DataFrame from the dictionary
        speed_angle_dataframe = pd.DataFrame(data_dict)
        speed_angle_dataframe.to_csv(os.path.join(data_fldr, save_prefix + '_speeds_and_angles.csv'), index=False)

    # Save the overall summary dataframe to CSV at the end of the outermost loop
    overall_summary_path = os.path.join(data_fldr, 'overall_summary.csv')
    overall_summary_dataframe.to_csv(overall_summary_path, index=False)
    return overall_summary_path


if __name__ == "__main__":
    analysis_logic(r'D:\OneDrive\Roger and Rozanne\spheroid analysis\Expt18 images to quantify\3D static\masked'
                   , {'experiment #': 18, 'condition': 'static'}, quantify_progress_print)