
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
import os
from constants import *


class BinarizedImage:
    """
    BinarizedImage loads a grayscale image, binarizes it using a specified threshold, and
    sets up the path for saving outputs.

    Args:
        raw_image_path (str): File path of the raw image.
        save_path (str, optional): Directory path to save processed images. If None, uses the
                                   directory of the raw image.
        threshold (int, optional): Threshold value for binarization. Defaults to 36.

    Attributes:
        img_path (str): Path of the raw image.
        img_name (str): Name of the image file.
        img_ext (str): Extension of the image file.
        save_fldr_path (str): Directory to save processed images.
        grayscale_array (ndarray): Loaded grayscale image.
        binary_array (ndarray): Binarized version of the grayscale image.
        threshold (int): Used threshold value.
        contours (list): List of contours found in the image.
        last_guassian_kernel (tuple): Last used Gaussian kernel for blurring.
    """

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
        """
        This method updates the binary array based on a new threshold and an optional boundary. If
        a boundary is provided, the threshold is applied only within that boundary.

        Args:
            threshold (int): New threshold value for binarization.
            boundary (list of tuples, optional): Points defining the boundary within which to apply
                                                 the threshold. If None, applies to the whole image.
        """

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
        """
        This method applies Gaussian blurring (if a kernel is provided) before finding contours.
        The contours are sorted by area in descending order.

        Args:
            guassian_kernel (tuple, optional): Kernel size for Gaussian blurring. If None, blurring
                                               is skipped.
        """
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
        """
        Calculate the centroid of the spheroid (binarized image).

        Computes the center of mass of the binarized image, returning the X and Y coordinates.

        Returns:
            tuple: The (x, y) coordinates of the centroid.
        """
        # Calculate the center of mass of the binarized image
        M = cv2.moments(self.binary_array.astype(np.uint8))
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = self.grayscale_array.shape[1] // 2, self.grayscale_array.shape[0] // 2

        return cX, cY

    def create_circular_mask(self):
        """
        Create and apply a circular mask centered at the spheroid centroid. The radius of the circle is the smallest
        distance from the centroid to the image edges. This mask is then applied to the binary array.
        """

        # Find the centroid
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
        """
        Apply a higher threshold to all contours except the largest one.

        This method is used to further refine the binarization by setting a high threshold value
        (256) to smaller contours.
        """

        if self.contours:
            # Create a mask for all contours except the largest one
            mask = np.zeros_like(self.grayscale_array, dtype=np.uint8)
            for contour in self.contours[1:]:
                cv2.fillPoly(mask, [contour], True)
            # Apply the local threshold to the mask
            self.binary_array[mask] = 256

    def save_binarized_image(self):
        """
        Save the binarized image to the designated folder. The method saves two versions of the image: one with and one
        without the circular mask. The images are named based on the threshold and kernel size used.
        """
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