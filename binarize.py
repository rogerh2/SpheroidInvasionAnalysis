
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
import os
from constants import *

# Define the BinarizedImage class with required functionalities
class BinarizedImage:

    def __init__(self, raw_image_path, save_path=None, threshold=36):
        self.img_path = raw_image_path
        file_name = os.path.basename(raw_image_path)
        self.img_name, self.img_ext = os.path.splitext(file_name)

        # If no explicit save path save in the same directory as the images
        if save_path is None:
            self.save_fldr_path = os.path.dirname(raw_image_path)
        else:
            self.save_fldr_path = save_path

        # Load the raw image (assuming it's grayscale)
        self.grayscale_array = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        self.binary_array = self.grayscale_array >= int(threshold)
        self.threshold = threshold
        self.contours = []
        self.last_guassian_kernel = None

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
        # Store the last used Gaussian kernel in a class attribute
        self.last_guassian_kernel = guassian_kernel

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

    def find_spheroid_centroid(self):
        # Calculate the center of mass of the binarized image
        M = cv2.moments(self.binary_array.astype(np.uint8))
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = self.grayscale_array.shape[1] // 2, self.grayscale_array.shape[0] // 2

        return cX, cY

    def create_circular_mask(self):
        cX, cY = self.find_spheroid_centroid()

        # Calculate the largest radius possible before hitting the image edge
        radius = min(cX, cY, self.grayscale_array.shape[1]-cX, self.grayscale_array.shape[0]-cY)

        # Create a circular mask
        Y, X = np.ogrid[:self.grayscale_array.shape[0], :self.grayscale_array.shape[1]]
        dist_from_center = np.sqrt((X - cX)**2 + (Y-cY)**2)
        mask = dist_from_center <= radius

        # Apply the mask to the binary array
        self.binary_array = np.logical_and(self.binary_array, mask)

    def apply_contour_threshold(self):
        """Apply a local threshold of 256 to all contours except the largest."""
        if self.contours:
            # Create a mask for all contours except the largest one
            mask = np.zeros_like(self.grayscale_array, dtype=np.uint8)
            for contour in self.contours[1:]:
                cv2.fillPoly(mask, [contour], True)
            # Apply the local threshold to the mask
            self.binary_array[mask] = 256

    def save_binarized_image(self):
        if self.last_guassian_kernel is None:
            k_size = 0
        else:
            k_size = self.last_guassian_kernel[0]

        save_name = f'{self.img_name}_binarized_thresh-{self.threshold}_kernel-{k_size}'

        # Save the binarized image with the circular mask
        unmasked_image = Image.fromarray((self.binary_array * 255).astype(np.uint8))
        unmasked_image.save(os.path.join(self.save_fldr_path
                        , f'{save_name}_{UNMASKED}{self.img_ext}'))

        self.create_circular_mask()

        # Save the binarized image without the circular mask
        # TODO make a masked folder and unmasked folder and save each in their respective folder
        masked_image = Image.fromarray((self.binary_array * 255).astype(np.uint8))
        masked_image.save(os.path.join(self.save_fldr_path, f'{save_name}_{MASKED}{self.img_ext}'))