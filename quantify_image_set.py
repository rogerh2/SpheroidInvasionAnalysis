import os
import re
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial import cKDTree
from constants import *


class SpheroidImage:

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

        # Generate a range of numbers for the width and height
        x_range = np.arange(source_image.shape[0])
        y_range = np.arange(source_image.shape[1])

        # Create meshgrid of coordinates
        self.x_coords, self.y_coords = np.meshgrid(y_range, x_range)

    def center_boundary(self, boundary):
        # Calculate moments of the binary image for centering the contour
        M = cv2.moments(boundary)
        bX = int(M["m10"] / M["m00"])
        bY = int(M["m01"] / M["m00"])

        # Translate the contour to the center of the target image
        return boundary + self.centroid - np.array([bX, bY])

    def get_pixle_coor_outside_boundary(self, boundary):
        # Create a mask for the region inside the boundary

        mask = np.ones_like(self.img_array, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(boundary)], False)
        mask = mask.astype(bool)

        x_coor_outside_bound = self.x_coords[mask][self.img_array[mask] > 0]
        y_coor_outside_bound = self.y_coords[mask][self.img_array[mask] > 0]

        return np.stack((x_coor_outside_bound, y_coor_outside_bound), axis=1)

    def intersection_distance(self, boundary):
        # IntersectionDistance finds the minimum distance from the Day0 boundary to a point of interest (POI).
        # The POI is on a line between the spheroid centroid (centroid_loc, x0) and the invaded cell pixels (outer_pixels, x1).
        # The function zeros the coordinate system at x0. Each point should be arranged as point = [x_loc; y_loc]
        # Outerpixels is fed in as one {cell} at a time

        outer_pixels_full = self.get_pixle_coor_outside_boundary(boundary)

        batch_size = 10000
        num_pix = len(outer_pixels_full)
        num_batches = num_pix // batch_size + (num_pix % batch_size != 0)
        distance_magnitude = np.zeros(num_pix)
        close_inds = np.zeros(num_pix)

        for i in range(num_batches):
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

            # - Part 1: The half of the boundary that is on the opposite side from POI
            #  - not nearest the POI
            perp_x1 = 1
            perp_x2 = -1
            perp_y1 = (-1 / (m + 1e-8))
            perp_y2 = (1 / (m + 1e-8))
            perp_valb = (xb_reshaped - perp_x1) *  (perp_y2 - perp_y1) - (yb_reshaped - perp_y1) *  (perp_x2 - perp_x1)
            perp_val_locs = (x1_reshaped - perp_x1) *  (perp_y2 - perp_y1) - (y1_reshaped - perp_y1) *  (perp_x2 - perp_x1)
            perp_mask = np.not_equal((perp_valb  /  np.abs(perp_valb)),  (perp_val_locs  /  np.abs(perp_val_locs)))

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
            xb_close = xb[current_close_inds]  # N
            yb_close = yb[current_close_inds]  # N
            distance_magnitude[i * batch_size : min((i + 1) * batch_size, num_pix)] = np.sqrt((x1 - xb_close) ** 2 + (y1 - yb_close) ** 2)  # shape N

        return distance_magnitude, close_inds, outer_pixels_full

    def get_angles_outside_boundary(self, boundary):
        outer_pixels = self.get_pixle_coor_outside_boundary(boundary)

        centered_x_coor = outer_pixels[:, 0] - self.centroid[0]
        centered_x_coor = np.sign(centered_x_coor) * np.clip(np.abs(centered_x_coor), 1e-8, np.inf)
        centered_y_coor = outer_pixels[:, 1] - self.centroid[1]
        return np.arctan(centered_y_coor / centered_x_coor)





class QuantImageSet:

    def __init__(self, image_fpaths, save_path=None):
        sample_times = np.array([int(re.search(PATTERN, os.path.basename(filename)).group(1))
                        for filename in image_fpaths if re.search(PATTERN, filename)])
        array_paths = np.array(image_fpaths)

        self.times = np.sort(sample_times)
        self.paths = array_paths[sample_times.argsort()]
        self.images = [SpheroidImage(fpath) for fpath in self.paths]

        # If no explicit save path save in the same directory as the images
        if save_path is None:
            self.save_fldr_path = os.path.dirname(image_fpaths[0])
        else:
            self.save_fldr_path = save_path

    def distances_outside_initial_boundary(self):
        init_bound = self.images[0].boundary.squeeze()
        distances = []
        angles_ls = []
        indices = []
        coordinates = []

        for img in self.images[1:]:
            centered_boundary = img.center_boundary(init_bound)
            dist, inds, coor = img.intersection_distance(centered_boundary)
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
            # point_rx, = ax.plot(coor[0, 0], coor[0, 1], 'r.')  # Initial red x
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
            #     point_rx.set_data(current_pixles[::, 0], current_pixles[::, 1])
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
            #     start_x = center[0] - length * np.cos(angle_rad)
            #     start_y = center[1] - length * np.sin(angle_rad)
            #
            #     # Update the line and point on the plot
            #     line.set_xdata([start_x, end_x])
            #     line.set_ydata([start_y, end_y])
            #     point_rx.set_data(pix_pt[0], pix_pt[1])
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
            angles_ls.append(angles)

        return distances, indices, coordinates







if __name__ == "__main__":
    data_fldr = r'D:\OneDrive\Roger and Rozanne\spheroid analysis\Expt18 images to quantify\test'
    image_fpaths = os.listdir(data_fldr)
    process_masked = True

    processed_experiments = []

    for fname in image_fpaths:
        _, ext = os.path.splitext(fname)
        exp_num = int(fname.split('_')[0])
        day = int(re.search(PATTERN, fname).group(1))
        is_masked = fname.split('_')[-1][:-len(ext)] == MASKED

        # Check if this experiment was already processed or has the propper masking
        if (exp_num in processed_experiments) or (process_masked != is_masked):
            continue

        fpaths_for_this_experiment = []

        for filename in image_fpaths:
            if (int(filename.split('_')[0]) == exp_num) and (filename.split('_')[-1][:-len(ext)] == MASKED):
                fpaths_for_this_experiment.append(os.path.join(data_fldr, filename))

        image_set_for_this_experiment = QuantImageSet(fpaths_for_this_experiment)
        distances, indices, outer_pixles = image_set_for_this_experiment.distances_outside_initial_boundary()

    # TODO create plots