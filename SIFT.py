import numpy as np
import cv2 as cv
import math
from scipy import ndimage

class SIFT:
    #convert image to grayscale 
    def __init__(self, image):
        self.image = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float32)
    #this return the base blurred image
    def generate_base_image(self, image, sigma=1.6, assum_blur=0.5):
        image = cv.resize(image, (0, 0), fx=2, fy=2)
        sigma_diff = max(math.sqrt((sigma**2) - 2*(assum_blur**2)), 0.01)
        return cv.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

    def compute_number_of_octaves(self, image_shape):
        """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)"""
        return int(round(math.log(min(image_shape)) / math.log(2) - 1))

    def get_sigmas_per_octave(self, sigma, num_intervals):
        num_images_per_octave = num_intervals + 3
        k = 2 ** (1 / num_intervals)
        sigmas = np.zeros(num_images_per_octave)
        sigmas[0] = sigma

        for i in range(1, num_images_per_octave):
            prev_sigma = sigma * (k ** (i - 1))
            sigma_total = prev_sigma * k
            sigmas[i] = math.sqrt(sigma_total**2 - prev_sigma**2)

        return sigmas

    def generate_gaussian_images(self, base_image, num_octaves=4, num_intervals=3, sigma=1.6):
        sigmas = self.get_sigmas_per_octave(sigma, num_intervals)
        gaussian_images = []

        for octave_index in range(num_octaves):
            gaussian_images_per_octave = [base_image]
            for sigma in sigmas[1:]:
                image = cv.GaussianBlur(base_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
                gaussian_images_per_octave.append(image)

            gaussian_images.append(gaussian_images_per_octave)
            base_image = gaussian_images_per_octave[3]
            base_image = cv.resize(base_image, (base_image.shape[1] // 2, base_image.shape[0] // 2), interpolation=cv.INTER_NEAREST)

        return gaussian_images

    def generate_dog_images(self):
        base_image= self.generate_base_image(self.image)
        gaussian_pyramid= self.generate_gaussian_images(base_image)
        dog_pyramid = []

        for octave in gaussian_pyramid:
            dog_images = []
            for i in range(1, len(octave)):
                dog = cv.subtract(octave[i], octave[i - 1])
                dog_images.append(dog)
            dog_pyramid.append(dog_images)

        return dog_pyramid

    def find_scale_space_extrema(self, contrast_threshold=0.03):
        keypoints = []
        dog_pyramid= self.generate_dog_images()
        for octave_idx, octave in enumerate(dog_pyramid):
            for scale_idx in range(1, len(octave) - 1):  # avoid first and last for 3x3x3
                prev_img = octave[scale_idx - 1]
                curr_img = octave[scale_idx]
                next_img = octave[scale_idx + 1]

                height, width = curr_img.shape

                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        value = curr_img[y, x]
                        if abs(value) < contrast_threshold:
                            continue  # low contrast → reject early

                        patch_prev = prev_img[y - 1:y + 2, x - 1:x + 2]
                        patch_curr = curr_img[y - 1:y + 2, x - 1:x + 2]
                        patch_next = next_img[y - 1:y + 2, x - 1:x + 2]

                        # 3x3x3 cube flattened
                        cube = np.stack([patch_prev, patch_curr, patch_next]).flatten()
                        center_index = 13  # center of 3x3x3 = 27 values
                        center_value = cube[center_index]

                        # Remove center pixel for comparison
                        neighbors = np.delete(cube, center_index)

                        if center_value > np.max(neighbors) or center_value < np.min(neighbors):
                        # Refine the keypoint location
                            refined = self.refine_keypoint(dog_pyramid, octave_idx, scale_idx, x, y)
                        if refined[0] is None:
                            continue
                        refined_x, refined_y, refined_scale = refined

                        # Check if it's edge-like
                        if self.is_edge_like(curr_img, int(round(refined_x)), int(round(refined_y))):
                            continue  # discard edge-like keypoints

                        # Save the refined keypoint
                        keypoints.append((octave_idx, refined_scale, refined_x, refined_y))
        return keypoints

    def refine_keypoint(self, dog_pyramid, octave, scale, x, y):
        if (x < 1 or x >= dog_pyramid[octave][scale].shape[1] - 1 or
            y < 1 or y >= dog_pyramid[octave][scale].shape[0] - 1 or
            scale <= 0 or scale >= len(dog_pyramid[octave]) - 1):
            return None, None, None

        prev_img = dog_pyramid[octave][scale - 1]
        curr_img = dog_pyramid[octave][scale]
        next_img = dog_pyramid[octave][scale + 1]

        dx = (curr_img[y, x + 1] - curr_img[y, x - 1]) * 0.5
        dy = (curr_img[y + 1, x] - curr_img[y - 1, x]) * 0.5
        ds = (next_img[y, x] - prev_img[y, x]) * 0.5
        gradient = np.array([dx, dy, ds])

        dxx = curr_img[y, x + 1] - 2 * curr_img[y, x] + curr_img[y, x - 1]
        dyy = curr_img[y + 1, x] - 2 * curr_img[y, x] + curr_img[y - 1, x]
        dss = next_img[y, x] - 2 * curr_img[y, x] + prev_img[y, x]

        dxy = ((curr_img[y + 1, x + 1] - curr_img[y + 1, x - 1]) -
               (curr_img[y - 1, x + 1] - curr_img[y - 1, x - 1])) * 0.25
        dxs = ((next_img[y, x + 1] - next_img[y, x - 1]) -
               (prev_img[y, x + 1] - prev_img[y, x - 1])) * 0.25
        dys = ((next_img[y + 1, x] - next_img[y - 1, x]) -
               (prev_img[y + 1, x] - prev_img[y - 1, x])) * 0.25

        hessian = np.array([
            [dxx, dxy, dxs],
            [dxy, dyy, dys],
            [dxs, dys, dss]
        ])

        try:
            offset = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if np.any(np.abs(offset) > 1.5):
                return None, None, None
        except np.linalg.LinAlgError:
            return None, None, None

        contrast = curr_img[y, x] + 0.5 * np.dot(gradient, offset)
        if abs(contrast) < 0.03:
            return None, None, None

        refined_x = x + offset[0]
        refined_y = y + offset[1]
        refined_scale = scale + offset[2]

        return refined_x, refined_y, refined_scale

    def is_edge_like(self, curr_img, x, y, edge_threshold=10):
        dxx = curr_img[y, x + 1] - 2 * curr_img[y, x] + curr_img[y, x - 1]
        dyy = curr_img[y + 1, x] - 2 * curr_img[y, x] + curr_img[y - 1, x]
        dxy = ((curr_img[y + 1, x + 1] - curr_img[y + 1, x - 1]) -
               (curr_img[y - 1, x + 1] - curr_img[y - 1, x - 1])) * 0.25

        trace = dxx + dyy
        det = dxx * dyy - dxy ** 2

        if det <= 0:
            return True

        ratio = (trace ** 2) / det
        return ratio >= ((edge_threshold + 1) ** 2) / edge_threshold

    def assign_orientation(self, radius=8, num_bins=36):
        image=self.image
        keypoints= self.find_scale_space_extrema()
        oriented_keypoints = []
        for x, y in keypoints:
            x, y = int(x), int(y)
            if y < radius or y >= image.shape[0] - radius or x < radius or x >= image.shape[1] - radius:
                continue
            region = image[y - radius:y + radius + 1, x - radius:x + radius + 1]
            gx = ndimage.sobel(region, axis=1)
            gy = ndimage.sobel(region, axis=0)
            magnitude = np.hypot(gx, gy)
            orientation = np.rad2deg(np.arctan2(gy, gx)) % 360

            hist, _ = np.histogram(orientation, bins=num_bins, range=(0, 360), weights=magnitude)
            dominant_orientation = np.argmax(hist) * (360 // num_bins)

            oriented_keypoints.append((x, y, dominant_orientation))
        return oriented_keypoints


    def compute_descriptors(self, window_size=16, num_subregions=4, num_bins=8):
        image=self.image
        oriented_keypoints= self.assign_orientation()
        descriptors = []
        for x, y, orientation in oriented_keypoints:
            x, y = int(x), int(y)
            half_size = window_size // 2
            if y < half_size or y >= image.shape[0] - half_size or x < half_size or x >= image.shape[1] - half_size:
                continue

            region = image[y - half_size:y + half_size, x - half_size:x + half_size]
            gx = ndimage.sobel(region, axis=1)
            gy = ndimage.sobel(region, axis=0)
            magnitude = np.hypot(gx, gy)
            angle = (np.rad2deg(np.arctan2(gy, gx)) - orientation) % 360

            step = window_size // num_subregions
            descriptor = []
            for i in range(num_subregions):
                for j in range(num_subregions):
                    sub_mag = magnitude[i * step:(i + 1) * step, j * step:(j + 1) * step]
                    sub_ang = angle[i * step:(i + 1) * step, j * step:(j + 1) * step]
                    hist, _ = np.histogram(sub_ang, bins=num_bins, range=(0, 360), weights=sub_mag)
                    descriptor.extend(hist)
            descriptor = np.array(descriptor)
            descriptor /= np.linalg.norm(descriptor) + 1e-7
            descriptors.append(descriptor)
        return descriptors



