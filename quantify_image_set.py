import os
from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from constants import *
from queue import Queue
from matplotlib.backends.backend_pdf import PdfPages


def safe_for_divide(x, eps=1e-8):
    x_sign = np.sign(x)
    x_sign[x_sign == 0] = 1
    return x_sign * np.clip(np.abs(x), eps, np.inf)


class SpheroidImage:
    """
    Class representing a binarized spheroid image.

    Args:
        fpath (str): The file path to the spheroid image.
        kill_Q (queue.Queue): A queue to send a kill signal to end the analysis early if desired
        time_unit (str): The unit of time used for the images
        font_spec (dict): A dictionary defining the font size and font name for plot lables and title
        tick_size (int): A parameter controlling the size of the x and y lables / ticks
        batch_size (int): The number of pixels to process in one batch during the analysis

    Attributes:
        boundary (array): The boundary coordinates of the largest spheroid in the image
        centroid (array): The x y coordinates for the center of the boundary
        img_array (array): The spheroid image stored in array form
        x_coords (array): The x coordinates of every pixel in img_array
        y_coords (array): The y coordinates of every pixel in img_array
        batch_size (int): The number of pixels to process in one batch during the analysis
        fname (str): The stem of the file name of the spheroid image.
        kill_queue (queue.Queue): The queue to send a kill signal to end the analysis early if desired.
        time_unit (str): The unit of time used for the images.
        font_spec (dict): The font specifications for plot labels and title.
        tick_size (int): The size of the x and y labels / ticks.
    """

    def __init__(self, fpath, kill_Q: Queue, time_unit, font_spec, tick_size, batch_size):
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
        self.kill_queue = kill_Q
        self.time_unit = time_unit
        self.batch_size = batch_size

        # Generate a range of numbers for the width and height
        x_range = np.arange(source_image.shape[0])
        y_range = np.arange(source_image.shape[1])

        # Create meshgrid of coordinates
        self.x_coords, self.y_coords = np.meshgrid(y_range, x_range)
        self.font_spec = font_spec
        self.tick_size = tick_size

    def center_boundary(self, boundary, boundary_centroid):
        """
        Centers the input boundary with the spheroid centroid

        Args:
            boundary (array-like): The coordinates defining the boundary of the spheroid object.
            boundary_centroid (array-like): The centroid of the boundary

        Returns:
            array: The centered boundary
        """

        # Translate the contour to the center of the target image
        return boundary + self.centroid - boundary_centroid

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
        batch_size = self.batch_size
        num_pix = len(outer_pixels_full)
        num_batches = num_pix // batch_size + (num_pix % batch_size != 0)

        # Setup initial arrays to fill with data each batch
        distance_magnitude = np.zeros(num_pix)
        close_inds = np.zeros(num_pix)
        boundary_pixels_full = np.zeros(outer_pixels_full.shape)
        kill_sig = False

        # Convert the relevant arrays to float32 before the loop
        outer_pixels_full = outer_pixels_full.astype(np.float32)
        boundary = boundary.astype(np.float32)
        distance_magnitude = distance_magnitude.astype(np.float32)
        close_inds = close_inds.astype(np.float32)
        boundary_pixels_full = boundary_pixels_full.astype(np.float32)

        # Process in float32 within the loop
        for i in range(num_batches):
            if not self.kill_queue.empty():
                kill_sig = self.kill_queue.get()
                break
            # Select the outer pixels for this batch
            outer_pixels = outer_pixels_full[i * batch_size: min((i + 1) * batch_size, num_pix)]

            # Define coordinates and center everything along the centroid location to make calculations easier
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
            m = y1_reshaped / safe_for_divide(x1_reshaped)  # shape M, N

            # Calculate the minimum distance from each boundary point to each line
            x_min = (xb_reshaped + m * yb_reshaped) / (1 + m ** 2)  # shape M, N
            y_min = m * x_min  # shape M, N
            d = np.sqrt((xb_reshaped - x_min) ** 2 + (yb_reshaped - y_min) ** 2)  # shape M, N

            # Filter out the parts of the boundary that are unwanted --------------- #

            # - Part 1: The half of the boundary that is on the opposite side from the pixel of interest (POI)
            #  - not nearest the POI
            perp_x1 = 1
            perp_x2 = -1
            perp_y1 = (-1 / safe_for_divide(m))
            perp_y2 = (1 / safe_for_divide(m))
            perp_valb = (xb_reshaped - perp_x1) * (perp_y2 - perp_y1) - (yb_reshaped - perp_y1) * (perp_x2 - perp_x1)
            perp_val_locs = (x1_reshaped - perp_x1) * (perp_y2 - perp_y1) - (y1_reshaped - perp_y1) * (
                        perp_x2 - perp_x1)

            perp_mask = np.not_equal((perp_valb / safe_for_divide(np.abs(perp_valb))),
                                     (perp_val_locs / safe_for_divide(np.abs(perp_val_locs))))

            # - Part 2: Filters out parts of boundary that are overlapping/ further from
            # - the centroid than the POI
            dist_mask = (xb_reshaped ** 2 + yb_reshaped ** 2) > (x1_reshaped ** 2 + y1_reshaped ** 2)

            # - Part 3: Apply filters
            tot_mask = np.logical_or(perp_mask, dist_mask)
            d[tot_mask] = np.max(d)

            # -------------------------------------------------------------------- #

            # Find the intercept by finding the boundary point closest to the line
            current_close_inds = np.argmin(d, axis=0)  # shape will be (N,)
            close_inds[i * batch_size: min((i + 1) * batch_size, num_pix)] = current_close_inds

            # calculate the distance magnitude
            xb_close = xb[current_close_inds]  # N
            yb_close = yb[current_close_inds]  # N
            distance_magnitude[i * batch_size: min((i + 1) * batch_size, num_pix)] = np.sqrt(
                (x1 - xb_close) ** 2 + (y1 - yb_close) ** 2)  # shape N
            boundary_pixels_full[i * batch_size: min((i + 1) * batch_size, num_pix)] = np.stack((xb_close, yb_close),
                                                                                                axis=1)

        # Convert arrays back to float64 at the end
        distance_magnitude = distance_magnitude.astype(np.float64)
        close_inds = close_inds.astype(np.float64)
        boundary_pixels_full = boundary_pixels_full.astype(np.float64)
        outer_pixels_full = outer_pixels_full.astype(np.float64)

        return distance_magnitude, close_inds, outer_pixels_full - self.centroid, boundary_pixels_full, kill_sig

    def pca(self, save_fldr_path, angles, speeds, t, save_images_to_pdf):
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
            save_images_to_pdf (bool): A boolean determining whether to save data as images in a folder or inside a pdf

        Returns:
            tuple: Contains principal angles, speeds, speed difference, and moments of inertia.
        """

        # Convert the relevant arrays to float32 before calculations
        angles = angles.astype(np.float32)
        speeds = speeds.astype(np.float32)

        if save_images_to_pdf:
            # Create a PDF file for saving the plots
            pdf_pages = PdfPages(os.path.join(save_fldr_path, 'pca_plots.pdf'))

        # Convert polar coordinates (angle, speed) to Cartesian coordinates (x, y)
        velocities = speeds[::, np.newaxis] * np.stack((np.cos(angles), np.sin(angles)), axis=1)

        # Center velocities about 0 so rotated and non rotated plots appear consistent
        velocities = velocities - velocities.mean(axis=0, keepdims=True)
        speeds = np.sqrt(velocities[:, 1] ** 2 + velocities[:, 0] ** 2)
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])

        # Initialize PCA and fit it to the velocities
        pca = PCA()
        score = pca.fit_transform(velocities)
        coeff = pca.components_

        # Transform the angles and speeds using the PCA scores
        transformed_angles = np.arctan2(score[:, 1], score[:, 0])
        transformed_speeds = np.sqrt(score[:, 1] ** 2 + score[:, 0] ** 2)

        # Define a function to plot an axis line at a given angle
        def plot_axis(ax, angle, max_radius):
            angle_rad = np.deg2rad(angle)  # Convert the angle to radians
            ax.plot([angle_rad, angle_rad], [0, max_radius], 'b')  # Plot in the positive direction
            ax.plot([angle_rad, angle_rad], [0, -max_radius], 'b')  # Plot in the negative direction as well

        # Calculate principal angles and speeds
        prin_angle1 = np.rad2deg(np.arctan2(coeff[0, 1], coeff[0, 0])) % 360
        prin_angle2 = np.rad2deg(np.arctan2(coeff[1, 1], coeff[1, 0])) % 360
        prin_speed1 = np.mean(np.abs(score[:, 0]))
        prin_speed2 = np.mean(np.abs(score[:, 1]))

        principle_angles = np.stack((prin_angle1, prin_angle2))
        principle_speeds = np.stack((prin_speed1, prin_speed2))

        # Plot original data
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.scatter(angles, speeds, marker='.', s=1)

        ax.set_title(f'Persistence speed (µm/{self.time_unit}) vs migration angle', **self.font_spec)
        plt.gca().tick_params(labelsize=self.tick_size)

        # Annotate plot with PCA information
        principle_angles_str = ', '.join(map(str, principle_angles))
        principle_speeds_str = ', '.join(map(str, principle_speeds))

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
        if save_images_to_pdf:
            pdf_pages.savefig()
        else:
            plt.savefig(os.path.join(save_fldr_path, f'{self.time_unit}{t}_img-i_speed_vs_angle_with_principle_components.png'))
        plt.close()

        # Plot transformed data
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.scatter(transformed_angles, transformed_speeds, marker='.', s=1)
        ax.set_title(f'Transformed persistence speed (µm/{self.time_unit}) vs migration angle', **self.font_spec)
        plt.gca().tick_params(labelsize=self.tick_size)

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
        if save_images_to_pdf:
            pdf_pages.savefig()
        else:
            plt.savefig(os.path.join(save_fldr_path, f'{self.time_unit}{t}_img-j_principle_coordinates_speed_vs_angle.png'))
        plt.close()

        # Calculate the difference between principal speeds
        prin_speed_difference = principle_speeds[0] - principle_speeds[1]

        # Calculate distances from boundary in pixels
        # Multiplying by time factor (t * 24 * 60) to get the distance
        transformed_outerdistance_lengths = transformed_speeds * (t * 24 * 60)
        transformed_outerdistances_xy = score * (t * 24 * 60)
    
        # Calculate moments of inertia from boundary
        # Assuming area for calculation = 1 for simplicity
        a = 1
        principle_Irb = np.sum(transformed_outerdistance_lengths ** 2 * a)
        principle_Ixb = np.sum(transformed_outerdistances_xy[:, 1] ** 2 * a)
        principle_Iyb = np.sum(transformed_outerdistances_xy[:, 0] ** 2 * a)

        # Close the PDF file
        if save_images_to_pdf:
            pdf_pages.close()

        # Convert back to float64 before returning results
        principle_angles = principle_angles.astype(np.float64)
        principle_speeds = principle_speeds.astype(np.float64)
        prin_speed_difference = prin_speed_difference.astype(np.float64)
        principle_Irb = principle_Irb.astype(np.float64)
        principle_Ixb = principle_Ixb.astype(np.float64)
        principle_Iyb = principle_Iyb.astype(np.float64)
        transformed_angles = transformed_angles.astype(np.float64)
        transformed_speeds = transformed_speeds.astype(np.float64)

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
        #centered_x_coor = np.sign(centered_x_coor) * np.clip(np.abs(centered_x_coor), 1e-8, np.inf)

        # Center the y-coordinates of the outer pixels relative to the centroid's y-coordinate
        centered_y_coor = outer_pixels[:, 1] - self.centroid[1]

        # Calculate and return the angles in radians using arctan2 for each outer pixel
        return np.arctan2(-centered_y_coor, centered_x_coor)


class QuantSpheroidSet:
    """
    This class processes a set of images corresponding to a single spheroid at multiple time points, sorts them based
    on time extracted from their filenames, and loads the images. It also sets the save folder path for any outputs.

    Args:
        image_fpaths (list): A list of file paths for the images.
        pattern (str): The regular expression that yields the time unit from the file name
        kill_Q (queue.Queue): A queue to send a kill signal to end the analysis early if desired
        time_unit (str): The unit of time used for the images
        pixel_size (float): The scale in micron / pixel for the microscope pixel_size
        font_spec (dict): A dictionary defining the font size and font name for plot lables and title
        tick_size (int): A parameter controlling the size of the x and y lables / ticks
        batch_size (int): The number of pixels to be processed in each batch in the analysis loop
        save_path (str, optional): The path where outputs should be saved. If None, uses the
                                   directory of the first image.

    Attributes:
        times (array): Sorted times extracted from image file names.
        paths (array): Image file paths sorted according to their corresponding times.
        images (list): A list of SpheroidImage objects loaded from the paths.
        save_fldr_path (str): Path to the folder where outputs will be saved.
        pixel_size (float): The scale in micron/pixel for the microscope pixel size.
        line_width (int): The width of lines in plots.
        marker_size (int): The size of markers in plots.
        font_spec (dict): The font specifications for plot labels and title.
        tick_size (int): The size of the x and y labels / ticks.
        time_str (str): The unit of time used for the images.
    """

    def __init__(self, image_fpaths, pattern, kill_Q: Queue, time_unit: str, pixel_size, font_spec, tick_size, batch_size, save_path=None):
        # Extracting time information from the image filenames
        sample_times = np.array([int(re.search(pattern, os.path.basename(filename)).group(1))
                        for filename in image_fpaths if re.search(pattern, filename)])
        array_paths = np.array(image_fpaths)

        # Sorting the times and paths based on the extracted times
        self.times = np.sort(sample_times)
        self.paths = array_paths[sample_times.argsort()]
        self.pixel_size = pixel_size

        # Loading images as SpheroidImage objects
        self.images = [SpheroidImage(fpath, kill_Q, time_unit, font_spec, tick_size, batch_size) for fpath in self.paths]
        self.line_width = 1
        self.marker_size = 10
        self.font_spec = font_spec
        self.tick_size = tick_size

        # If no explicit save path save in the same directory as the images
        if save_path is None:
            self.save_fldr_path = os.path.dirname(image_fpaths[0])
        else:
            self.save_fldr_path = save_path

        self.time_str = time_unit

    def distances_outside_initial_boundary(self, save_fldr, save_images_to_pdf):
        """
        Calculate distances to the pixels outside the initial boundary for each image in the
        set, excluding the first image.

        Args:
            save_fldr (str): The file path where the plots will be saved.
            save_images_to_pdf (bool): A boolean determining whether to save data as images in a folder or inside a pdf

        Returns:
            tuple: arrays of distances, indices, coordinates, angles, and outer coordinates.
        """

        if save_images_to_pdf:
            # Create a PDF file for saving the plots
            pdf_pages = PdfPages(os.path.join(save_fldr, 'intermediary_plots.pdf'))

        # Initial boundary is taken from the first image
        init_bound = self.images[0].boundary.squeeze()

        # Calculate moments of the binary image for centering the contour
        M = cv2.moments(init_bound)
        bX = int(M["m10"] / M["m00"])
        bY = int(M["m01"] / M["m00"])
        boundary_centroid = np.array([bX, bY])

        # Plot the binarized image and the boundary
        plt.imshow(self.images[0].img_array, cmap='gray')
        plt.plot(init_bound[:, 0], init_bound[:, 1], color='limegreen',
                 linewidth=self.line_width)  # Plot boundary in green
        plt.title('Binarized Image with Initial Boundary', **self.font_spec)
        plt.gca().tick_params(labelsize=self.tick_size)

        if save_images_to_pdf:
            pdf_pages.savefig()
        else:
            plt.savefig(os.path.join(save_fldr, f'{self.time_str}0_img-a_binarized_with_boundary.png'))

        plt.figure()
        plt.imshow(self.images[0].img_array, cmap='gray')
        plt.scatter(boundary_centroid[0], boundary_centroid[1], color='blue', marker='*', s=self.marker_size)
        plt.title('Initial Image with Boundary Centroid', **self.font_spec)
        plt.gca().tick_params(labelsize=self.tick_size)

        if save_images_to_pdf:
            pdf_pages.savefig()
        else:
            plt.savefig(os.path.join(save_fldr, f'{self.time_str}0_img-b_binarized_with_centroid.png'))

        # Initializing lists to store calculated values
        distances = []
        angles_ls = []
        indices = []
        coordinates = []
        outer_coordinates = []
        times = []
        kill_sig = False

        # Iterating over images (excluding the first) to calculate metrics
        for i, img in enumerate(self.images[1:], start=1):
            centered_boundary = img.center_boundary(init_bound, boundary_centroid)
            t = self.times[i]
            dist, inds, coor, boundary_coor, kill_sig = img.intersection_distance(centered_boundary)
            if kill_sig:
                break
            angles = img.get_angles_outside_boundary(centered_boundary)

            plt.figure()
            plt.imshow(img.img_array, cmap='gray')
            plt.scatter(img.centroid[0], img.centroid[1], color='blue', marker='*', s=self.marker_size)
            plt.title(f'{self.time_str}{t}: Boundary Centroid', **self.font_spec)
            plt.gca().tick_params(labelsize=self.tick_size)

            if save_images_to_pdf:
                pdf_pages.savefig()
            else:
                plt.savefig(os.path.join(save_fldr, f'{self.time_str}{t}_img-c_binarized_with_boundary.png'))
            plt.close()

            # Plot img.img_array with init_bound and centroids
            plt.figure()
            plt.imshow(img.img_array, cmap='gray')
            plt.plot(init_bound[:, 0], init_bound[:, 1], color='limegreen', linewidth=self.line_width)
            plt.scatter(boundary_centroid[0], boundary_centroid[1], color='limegreen', marker='*',
                        s=self.marker_size)  # Green star for boundary_centroid
            plt.scatter(img.centroid[0], img.centroid[1], color='blue', marker='*',
                        s=self.marker_size)  # Blue star for img.centroid
            plt.title(f'{self.time_str}{t}: Uncentered boundary', **self.font_spec)
            plt.gca().tick_params(labelsize=self.tick_size)
            if save_images_to_pdf:
                pdf_pages.savefig()
            else:
                plt.savefig(os.path.join(save_fldr, f'{self.time_str}{t}_img-d_uncentered_boundary.png'))
            plt.close()

            # Plot img.img_array with centered_boundary
            plt.figure()
            plt.imshow(img.img_array, cmap='gray')
            plt.plot(centered_boundary[:, 0], centered_boundary[:, 1], color='limegreen', linewidth=self.line_width)
            plt.scatter(img.centroid[0], img.centroid[1], color='blue', marker='*',
                        s=self.marker_size)  # Blue star for img.centroid
            plt.title(f'{self.time_str}{t}: Centered boundary', **self.font_spec)
            plt.gca().tick_params(labelsize=self.tick_size)
            if save_images_to_pdf:
                pdf_pages.savefig()
            else:
                plt.savefig(os.path.join(save_fldr, f'{self.time_str}{t}_img-e_centered_boundary.png'))
            plt.close()

            # Plot img.img_array with points in coor + img.centroid
            plt.figure()
            plt.imshow(img.img_array, cmap='gray')
            plt.plot(centered_boundary[:, 0], centered_boundary[:, 1], color='limegreen', linewidth=self.line_width)
            plt.scatter(coor[:, 0] + img.centroid[0], coor[:, 1] + img.centroid[1], marker='.', color='cyan',
                        s=1)
            plt.scatter(img.centroid[0], img.centroid[1], color='limegreen', marker='*',
                        s=self.marker_size)  # Blue star for img.centroid
            plt.title(f'{self.time_str}{t}: Centered boundary and outer pixels', **self.font_spec)
            plt.gca().tick_params(labelsize=self.tick_size)
            if save_images_to_pdf:
                pdf_pages.savefig()
            else:
                plt.savefig(os.path.join(save_fldr, f'{self.time_str}{t}_img-f_centered_boundary_outer_pixels.png'))
            plt.close()

            # Plot img.img_array with init_bound and lines between random points
            plt.figure()
            plt.imshow(img.img_array, cmap='gray')
            N = 300
            random_boundary_sample = np.random.choice(boundary_coor.shape[0], N, replace=False)
            sample_boundary_coor = boundary_coor[random_boundary_sample, :] + img.centroid
            sample_coor = coor[random_boundary_sample, :] + img.centroid
            for p1, p2 in zip(sample_boundary_coor, sample_coor):
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='cyan',
                         linewidth=self.line_width)  # Blue lines between points

            plt.scatter(img.centroid[0], img.centroid[1], color='blue', marker='*',
                        s=self.marker_size)  # Blue star for img.centroid
            plt.plot(centered_boundary[:, 0], centered_boundary[:, 1], color='blue', linewidth=self.line_width)
            plt.axhline(y=img.centroid[1], color='gray', linestyle='--')
            plt.title(f'{self.time_str}{t}: Distance from boundary', **self.font_spec)
            plt.gca().tick_params(labelsize=self.tick_size)
            if save_images_to_pdf:
                pdf_pages.savefig()
            else:
                plt.savefig(os.path.join(save_fldr, f'{self.time_str}{t}_distance_from_boundary.png'))
            plt.close()

            plt.figure()
            plt.imshow(img.img_array, cmap='gray')
            for p2 in sample_coor:
                plt.plot([img.centroid[0], p2[0]], [img.centroid[1], p2[1]], color='blue',
                         linewidth=self.line_width)  # Blue lines between points

            plt.plot(centered_boundary[:, 0], centered_boundary[:, 1], color='limegreen', linewidth=self.line_width)
            plt.axhline(y=img.centroid[1], color='gray', linestyle='--')

            plt.scatter(img.centroid[0], img.centroid[1], color='limegreen', marker='*',
                        s=self.marker_size, zorder=N + 10)  # Green star for img.centroid
            plt.title(f'{self.time_str}{t}: Distance from centroid', **self.font_spec)
            plt.gca().tick_params(labelsize=self.tick_size)
            if save_images_to_pdf:
                pdf_pages.savefig()
            else:
                plt.savefig(os.path.join(save_fldr, f'{self.time_str}{t}_img-g_distance_from_centroid.png'))
            plt.close()

            plt.figure()
            plt.imshow(img.img_array, cmap='gray')

            # Generate random indices to plot angles at
            inds = np.random.choice(angles.shape[0], 5, replace=False)

            # Update the angle and point based on the new index
            angle_rad = angles[inds]
            pix_pt = coor[inds] + img.centroid

            # Create label angles that remove the inverted y-axis from imshow
            angles_rad_for_labels = np.arctan2(-coor[inds, 1], coor[inds, 0])

            # Calculate the distances from the centroid
            length = np.sqrt(np.sum((pix_pt - img.centroid) ** 2, axis=1))

            # Calculate the new end point of the line
            point_locs = np.stack([img.centroid[0] + length * np.cos(angle_rad),
                                   img.centroid[1] - length * np.sin(angle_rad)], axis=1)

            for p in point_locs:
                plt.plot([img.centroid[0], p[0]], [img.centroid[1], p[1]],
                         linewidth=self.line_width)  # Blue lines between points

            # Using a second loop to ensure the legend only labels the lines
            for p in pix_pt:
                plt.scatter(p[0], p[1], color='red', marker='*', s=self.marker_size)

            plt.axhline(y=img.centroid[1], color='red', linestyle='--')
            plt.plot(centered_boundary[:, 0], centered_boundary[:, 1], color='limegreen', linewidth=self.line_width)

            angle_deg = (np.rad2deg(angles_rad_for_labels).astype(int) + 360) % 360
            labels = [f'{self.time_str}{t}: {a}°' for a in angle_deg]
            plt.legend(labels, loc='upper right')
            plt.title(f'{self.time_str}{t}: Angles of migration', **self.font_spec)
            plt.gca().tick_params(labelsize=self.tick_size)
            if save_images_to_pdf:
                pdf_pages.savefig()
            else:
                plt.savefig(os.path.join(save_fldr, f'{self.time_str}{t}_img-h_angles_of_migration.png'))
            plt.close()

            distances.append(dist * self.pixel_size)
            indices.append(inds)
            coordinates.append(coor * self.pixel_size)
            outer_coordinates.append((coor - boundary_coor) * self.pixel_size)
            angles_ls.append(angles)
            times.append(img)

        # Close the PDF file
        if save_images_to_pdf:
            pdf_pages.close()

        return distances, indices, coordinates, angles_ls, outer_coordinates, kill_sig


def plot_moment_of_inertia(title, moments, font_spec, tick_size):
    categories = ['Ir', 'Ix', 'Iy']
    fig = plt.figure()
    bars = plt.bar(categories, moments)
    plt.title(title, **font_spec)
    plt.ylabel('Moment (pixels^4)', **font_spec)
    plt.gca().tick_params(labelsize=tick_size)

    # Adding values on top of the bars
    for bar, value in zip(bars, moments):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2e}',
                 ha='center', va='bottom')

    return fig


def PlotPixelDistancesandAngles(save_fldr_path, t, outerdistance_lengths, angles_array, outer_distances_xy, centerdistance_lengths,
                                    full_distances_xy, num_days, pixel_size, save_images_to_pdf, time_unit, font_spec, tick_size):
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
        save_images_to_pdf (bool): A boolean determining whether to save data as images in a folder or inside a pdf
        time_unit (str): A string denoting the unit of time used in the measurements
        font_spec (dict): A dictionary defining the font size and font name for plot lables and title
        tick_size (int): The size parameter for the x and y lables / ticks

    Returns:
        tuple: A tuple containing moments of inertia from the boundary (Irb, Ixb, Iyb),
               moments of inertia from the center (Irc, Ixc, Iyc), the arrays of outerdistance lengths, outer distances xy,
               centerdistance lengths, full distances xy, speed array, and speed array converted to distance units.
    """

    if save_images_to_pdf:
        # Create a PDF file for saving the plots
        pdf_pages = PdfPages(os.path.join(save_fldr_path, 'data_plots.pdf'))

    plt.figure()

    # Plot distances in a histogram
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot

    h1, bins, _ = plt.hist(outerdistance_lengths, bins=20, rwidth=0.8)  # Histogram with 20 bins
    plt.title('Distances from boundary histogram', **font_spec)
    plt.xlabel('distance (µm)', **font_spec)
    plt.ylabel('frequency', **font_spec)
    plt.gca().tick_params(labelsize=tick_size)

    # get the current x limits
    x_min, x_max = plt.xlim()

    # Plot the cumulative values in a histogram
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
    cumDistValues = np.cumsum(h1)  # Cumulative sum of the histogram values
    plt.bar((bins[1::] + bins[0:-1]) / 2, cumDistValues, width=0.8 * (bins[1::] - bins[0:-1]))  # Bar plot
    plt.title('Cumulative distances from boundary histogram', **font_spec)
    plt.xlabel('distance (µm)', **font_spec)
    plt.ylabel('frequency', **font_spec)
    plt.gca().tick_params(labelsize=tick_size)

    # Set the x limits of the two subplots to the same value
    x_min, x_max = bins[0], bins[-1]

    plt.tight_layout()  # Adjust subplots to fit into figure area.
    if save_images_to_pdf:
        pdf_pages.savefig()
    else:
        plt.savefig(os.path.join(save_fldr_path, f'{time_unit}{t}_img-k_img-h_distances_histogram.png'))  # Save the figure
    plt.close()

    # Assuming 'angles' is a list of numpy arrays or lists
    #angles_array = np.concatenate(angles)  # Concatenate all angle arrays

    # Create a polar histogram
    plt.figure()
    ax = plt.subplot(111, polar=True)  # Create a polar subplot
    ax.hist(angles_array, bins=20, rwidth=0.9)  # Polar histogram with 20 bins
    plt.title('Angles histogram', **font_spec)
    plt.gca().tick_params(labelsize=tick_size)
    if save_images_to_pdf:
        pdf_pages.savefig()
    else:
        plt.savefig(os.path.join(save_fldr_path, f'{time_unit}{t}_img-l_angles_histogram.png'))
    plt.close()

    max_dist = np.max(outerdistance_lengths)
    median_dist = np.median(outerdistance_lengths)
    mean_dist = np.mean(outerdistance_lengths)

    # Plotting the bar graph
    categories = ['max', 'median', 'mean']
    values = [max_dist, median_dist, mean_dist]

    plt.figure()
    bars = plt.bar(categories, values)

    plt.title('Representative distance values from the boundary', **font_spec)
    plt.ylabel('distance (µm)', **font_spec)
    plt.gca().tick_params(labelsize=tick_size)

    # Adding values on top of the bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(np.ceil(value))}',
                 ha='center', va='bottom')

    if save_images_to_pdf:
        pdf_pages.savefig()
    else:
        plt.savefig(os.path.join(save_fldr_path, f'{time_unit}{t}_img-m_representative_distances.png'))
    plt.close()

    plt.figure()
    ax1 = plt.subplot(111, polar=True)
    ax1.scatter(angles_array, centerdistance_lengths, marker='.', s=1)
    plt.title("Distances from center (µm) vs angle", **font_spec)
    plt.gca().tick_params(labelsize=tick_size)
    if save_images_to_pdf:
        pdf_pages.savefig()
    else:
        plt.savefig(os.path.join(save_fldr_path, f'{time_unit}{t}_img-n_distances_vs_angle_center.png'))
    plt.close()

    # Plot the boundary distance vs angle values in a polar plot
    plt.figure()
    ax2 = plt.subplot(111, polar=True)
    ax2.scatter(angles_array, outerdistance_lengths, marker='.', s=1)
    plt.title("Distances from boundary (µm) vs angle", **font_spec)
    plt.gca().tick_params(labelsize=tick_size)
    if save_images_to_pdf:
        pdf_pages.savefig()
    else:
        plt.savefig(os.path.join(save_fldr_path, f'{time_unit}{t}_img-o_distances_vs_angle_boundary.png'))
    plt.close()

    # Convert distances to meters and calculate speed
    outerdistance_lengths_m = outerdistance_lengths * pixel_size
    centerdistance_lengths_m = centerdistance_lengths * pixel_size
    speed_array = outerdistance_lengths_m / num_days  # m/min

    # Plotting the speed vs angle
    plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.scatter(angles_array, speed_array, marker='.', s=1)
    plt.title(f'Speed (µm/{time_unit}) vs angle', **font_spec)
    plt.gca().tick_params(labelsize=tick_size)
    if save_images_to_pdf:
        pdf_pages.savefig()
    else:
        plt.savefig(os.path.join(save_fldr_path, f'{time_unit}{t}_img-p_speed_vs_angle.png'))
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

    # Plot the moment of inertia from center
    title = 'Moment of inertia from center'
    plot_moment_of_inertia(title, [Irc, Ixc, Iyc], font_spec, tick_size)

    if save_images_to_pdf:
        pdf_pages.savefig()
    else:
        plt.savefig(os.path.join(save_fldr_path, f'{time_unit}{t}_img-q_' + title.replace(' ', '_').lower() + '.png'))
    plt.close()

    # Plot the moment of inertia from boundary
    title = 'Moment of inertia from boundary'
    plot_moment_of_inertia(title, [Irb, Ixb, Iyb], font_spec, tick_size)

    if save_images_to_pdf:
        pdf_pages.savefig()
    else:
        plt.savefig(os.path.join(save_fldr_path, f'{time_unit}{t}_img-r_' + title.replace(' ', '_').lower() + '.png'))
    plt.close()

    if save_images_to_pdf:
        pdf_pages.close()

    return Irb, Ixb, Iyb, Irc, Ixc, Iyc, outerdistance_lengths, outer_distances_xy, centerdistance_lengths, full_distances_xy, speed_array, pixel_size * speed_array


def quantify_progress_print(progress):
        print(f'Quantifying data {progress}% complete')

def analysis_logic(data_fldr, master_id_dict, progress_print_fun, kill_queue: Queue, pattern, time_unit, pixel_scale
                   , font_spec, tick_size, batch_size, save_images_to_pdf=False):
    """
    Loops through spheroid images and saves the relevant data for further analysis. Groups spheroids by their prefix
    number and characterizes them based on the time points in the file name expressed as <time unit>T.

    Args:
        data_fldr (str): The file path where the images are stored and the data will be saved
        master_id_dict (dict): Dictionary containing meta data for this set of spheroid images
        progress_print_fun (callable): A function to display the analysis progress
        kill_queue (queue.Queue): A queue to send a kill signal to end the analysis early if desired
        pattern (str): A regular expression that finds the time point from the image name
        time_unit (str): The unit of measure for time
        pixel_scale (float): The scale in micron / pixel for the microscope pixel_size
        font_spec (dict): A dictionary defining the font size and font name for plot lables and title
        tick_size (int): The size parameter for the x and y lables / ticks
        batch_size (int): The number of pixels to process in one batch during the analysis
        save_images_to_pdf (bool): A boolean stating whether to save images to pdf (currently some pdf files are corrupted)
    """

    print('Analysis started')

    # Filtering out directories from image_fpaths
    image_fpaths = []

    for f in os.listdir(data_fldr):
        _, img_ext = os.path.splitext(f)

        if img_ext in ['.tif', '.png', '.jpg']:
            image_fpaths.append(f)

    if not len(image_fpaths):
        return

    processed_experiments = []

    overall_summary_dataframe = pd.DataFrame()

    for i, fname in enumerate(image_fpaths):

        # Update progress bar
        progress = 100 * i / len(image_fpaths)

        # Kill the program early if a kill signal is sent
        if not kill_queue.empty():
            kill_queue.get()
            print('early stop')
            return

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

        image_set_for_this_experiment = QuantSpheroidSet(fpaths_for_this_experiment, pattern, kill_queue, time_unit
                                                         , pixel_scale, font_spec, tick_size, batch_size)
        distances, indices, pixles, angles, outer_coordinates, kill_sig = image_set_for_this_experiment.distances_outside_initial_boundary(save_fldr_path, save_images_to_pdf)

        if kill_sig:
            print('early stop')
            return

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

        for j in range(0, len(image_set_for_this_experiment.images) - 1):
            # Kill the program early if a kill signal is sent
            if not kill_queue.empty():
                kill_queue.get()
                print('early stop')
                return

            speed_angle_columns_data = []
            pca_speed_angle_columns_data = []

            img, t = image_set_for_this_experiment.images[j + 1], image_set_for_this_experiment.times[j + 1]

            metrics = PlotPixelDistancesandAngles(save_fldr_path, t, distances[j], angles[j], outer_coordinates[j]
                                                  , np.sqrt(pixles[j][::, 0] ** 2 + pixles[j][::, 1] ** 2),
                                                  pixles[j], 2, 1, save_images_to_pdf, time_unit, font_spec, tick_size)
            Irb, Ixb, Iyb, Irc, Ixc, Iyc, outerdistance_lengths, outer_distances_xy, centerdistance_lengths \
                , full_distances_xy, speed_array, speed_dimensionalized = metrics

            fname_list.append(img.fname)
            areas.append(np.sum(img.img_array) * pixel_scale ** 2)
            Irb_values.append(Irb)
            Ixb_values.append(Ixb)
            Iyb_values.append(Iyb)
            Irc_values.append(Irc)
            Ixc_values.append(Ixc)
            Iyc_values.append(Iyc)
            max_speeds.append(np.max(speed_dimensionalized))
            mean_speeds.append(np.mean(speed_dimensionalized))
            median_speeds.append(np.median(speed_dimensionalized))
            max_angles.append(np.max(angles[j][:]))
            mean_angles.append(np.mean(angles[j][:]))
            median_angles.append(np.median(angles[j][:]))

            pca_metrics = img.pca(save_fldr_path, angles[j], speed_dimensionalized, t, save_images_to_pdf)
            pa0, pa1, ps0, ps1, prin_speed_difference, principle_Irb, principle_Ixb \
                , principle_Iyb, transformed_angles, transformed_speeds = pca_metrics

            # Append to the columns list with appropriate names
            speed_angle_columns_data.append((f'Speeds at {time_unit} {t}', speed_dimensionalized))
            speed_angle_columns_data.append((f'Angles at {time_unit} {t}', angles[j][:]))
            pca_speed_angle_columns_data.append((f'PCA transformed Speeds at {time_unit} {t}', transformed_speeds))
            pca_speed_angle_columns_data.append((f'PCA transformed Angles at {time_unit} {t}', transformed_angles))

            pa0_values.append(pa0)
            pa1_values.append(pa1)
            ps0_values.append(ps0)
            ps1_values.append(ps1)
            prin_speed_diff_values.append(prin_speed_difference)
            principle_Irb_values.append(principle_Irb)
            principle_Ixb_values.append(principle_Ixb)
            principle_Iyb_values.append(principle_Iyb)

            # Create the DataFrame from the dictionary
            speed_angle_dataframe = pd.DataFrame(dict(speed_angle_columns_data))
            speed_angle_dataframe.to_csv(os.path.join(save_fldr_path, save_prefix + f'_speeds_and_angles_{time_unit}{t}_full.csv'),
                                         index=False)

            # Create the DataFrame from the dictionary
            pca_speed_angle_dataframe = pd.DataFrame(dict(pca_speed_angle_columns_data))
            pca_speed_angle_dataframe.to_csv(os.path.join(save_fldr_path, save_prefix + f'_pca_speeds_and_angles_{time_unit}{t}_full.csv'),
                                             index=False)

        # Create a dictionary of the summary values
        summary_dict = {id_name: [id_value] * (len(image_set_for_this_experiment.images) - 1) for id_name, id_value
                        in id_dict.items()}

        summary_dict.update({
            'file': fname_list,
            f'{time_unit}': image_set_for_this_experiment.times[1:],
            f'{time_unit} 0 areas': A0 * np.ones(len(areas)),
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
            'max_principle_angle': pa0_values,
            'min_principle_angle': pa1_values,
            'max_principle_speed': ps0_values,
            'min_principle_speed': ps1_values,
            'principle_speed_difference': prin_speed_diff_values,
            'principle_Irb': principle_Irb_values,
            'min_principle_Ib': principle_Ixb_values,
            'max_principle_Ib': principle_Iyb_values
        })

        summary_dataframe = pd.DataFrame(summary_dict)
        # Concatenate current summary dataframe to overall summary dataframe
        overall_summary_dataframe = pd.concat([overall_summary_dataframe, summary_dataframe])

    # Save the overall summary dataframe to CSV at the end of the outermost loop
    overall_summary_path = os.path.normpath(os.path.join(data_fldr, 'overall_summary.csv'))
    overall_summary_dataframe.to_csv(overall_summary_path, index=False)
    return overall_summary_path


if __name__ == "__main__":
    # Run this after binarizing your images
    analysis_logic(r'.\Expt 19\3D dynamic\masked'
                   , {'experiment #': 19, 'condition': 'dynamic'}, quantify_progress_print, Queue(), PATTERN
                   , 'day', 1, FONT_SPEC, 11, batch_size=10000)