import cv2 as cv
import numpy as np
import math
from scipy import ndimage

class SIFT:
    def __init__(self, image):
        self.original_image = image
        self.image = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float32)

    def generate_base_image(self, image, sigma=1.6, assum_blur=0.5):
        image = cv.resize(image, (0, 0), fx=2, fy=2)
        sigma_diff = max(math.sqrt((sigma**2) - 2*(assum_blur**2)), 0.01)
        return cv.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

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

        for _ in range(num_octaves):
            gaussian_images_per_octave = [base_image]
            for sigma in sigmas[1:]:
                image = cv.GaussianBlur(base_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
                gaussian_images_per_octave.append(image)

            gaussian_images.append(gaussian_images_per_octave)
            base_image = gaussian_images_per_octave[3]
            base_image = cv.resize(base_image, (base_image.shape[1] // 2, base_image.shape[0] // 2), interpolation=cv.INTER_NEAREST)

        return gaussian_images

    def generate_dog_images(self):
        base_image = self.generate_base_image(self.image)
        gaussian_pyramid = self.generate_gaussian_images(base_image)
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
        dog_pyramid = self.generate_dog_images()
        for octave_idx, octave in enumerate(dog_pyramid):
            for scale_idx in range(1, len(octave) - 1):
                prev_img = octave[scale_idx - 1]
                curr_img = octave[scale_idx]
                next_img = octave[scale_idx + 1]

                height, width = curr_img.shape

                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        value = curr_img[y, x]
                        if abs(value) < contrast_threshold:
                            continue

                        patch_prev = prev_img[y - 1:y + 2, x - 1:x + 2]
                        patch_curr = curr_img[y - 1:y + 2, x - 1:x + 2]
                        patch_next = next_img[y - 1:y + 2, x - 1:x + 2]

                        cube = np.stack([patch_prev, patch_curr, patch_next]).flatten()
                        center_value = cube[13]
                        neighbors = np.delete(cube, 13)

                        if center_value > np.max(neighbors) or center_value < np.min(neighbors):
                            refined = self.refine_keypoint(dog_pyramid, octave_idx, scale_idx, x, y)
                            if refined is None or refined[0] is None:
                                continue
                            refined_x, refined_y, refined_scale = refined
                            if self.is_edge_like(curr_img, int(round(refined_x)), int(round(refined_y))):
                                continue
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
        if x < 1 or x >= curr_img.shape[1] - 1 or y < 1 or y >= curr_img.shape[0] - 1:
            return True
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

    def assign_orientation(self, keypoints, radius=8, num_bins=36):
        image = self.image
        oriented_keypoints = []
        for octave_idx, scale, x, y in keypoints:
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

    def compute_descriptors(self, keypoints, window_size=16, num_subregions=4, num_bins=8):
        image = self.image
        descriptors = []
        for x, y, orientation in keypoints:
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
            descriptor = np.clip(descriptor, 0, 0.2)
            descriptor /= np.linalg.norm(descriptor) + 1e-7
            descriptors.append(descriptor)

        return descriptors
