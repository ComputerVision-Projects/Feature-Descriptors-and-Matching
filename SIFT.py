import numpy as np
import cv2 as cv 
from cv2 import GaussianBlur, resize
import math
import numpy as np
class SIFT:

    def __init__(self,image):
        self.image = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float32)

    def generate_base_image(self,image,sigma=1.6,assum_blur=0.5):

        image=resize(image,(0,0),fx=2,fy=2) 

        sigma_diff=max(math.sqrt((sigma**2) - 2*(assum_blur**2)),0.01)

        return GaussianBlur(image,(0,0),sigmaX=sigma_diff,sigmaY=sigma_diff)
    
    def compute_number_of_octaves(self,image_shape):
        """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)"""
        return int(round(math.log(min(image_shape))/math.log(2)-1))


    def get_sigmas_per_octave(self,sigma, num_intervals):
        num_images_per_octave = num_intervals + 3
        k = 2 ** (1 / num_intervals)
        sigmas=np.zeros(num_images_per_octave)
        sigmas[0] = sigma
        
        for i in range(1, num_images_per_octave):
            # Compute the total sigma of the i-th image relative to the previous one
            prev_sigma = sigma * (k ** (i - 1))
            sigma_total=prev_sigma*k
            sigmas[i] = math.sqrt(sigma_total**2 - prev_sigma**2)
            
        return sigmas
    
    def generate_gaussian_images(self, base_image, num_octaves=4, num_intervals=3, sigma=1.6):
        # Step 1: Get the list of sigmas for one octave
        sigmas = self.get_sigmas_per_octave(sigma, num_intervals)  # returns list of sigmas

        # Step 2: Initialize empty list to store gaussian images per octave
        gaussian_images=[]

        # Step 3: Loop through octaves
        for octave_index in range(num_octaves):
            gaussian_images_per_octave = []  # images for this octave
            gaussian_images_per_octave.append(base_image)
            # Step 4: Blur the base image with increasing sigma values
            for sigma in sigmas[1:]:    
                    image = cv.GaussianBlur(base_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
                    gaussian_images_per_octave.append(image)
                    base_image=image

            gaussian_images.append(gaussian_images_per_octave)
            base_image=gaussian_images[-3]
            # Step 5: Downsample the image for the next octave
            base_image = cv.resize(base_image, (base_image.shape[1] // 2, base_image.shape[0] // 2), interpolation=cv.INTER_NEAREST)

        return gaussian_images
    
    def generate_dog_images(self, gaussian_pyramid):
        dog_pyramid = []

        for octave in gaussian_pyramid:
            dog_images=[]
            for i in range(1,len(octave)):
                dog=cv.subtract(octave[i],octave[i-1])
                dog_images.append(dog)
            dog_pyramid.append(dog_images)    

        return dog_pyramid
    

    def find_scale_space_extrema(self, dog_pyramid, contrast_threshold=0.03):
        keypoints = []

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
                            keypoints.append((octave_idx, scale_idx, x, y))

        return keypoints
    


    def refine_keypoint(dog_pyramid, octave, scale, x, y):
        # Boundary check
        if (x < 1 or x >= dog_pyramid[octave][scale].shape[1] - 1 or
            y < 1 or y >= dog_pyramid[octave][scale].shape[0] - 1 or
            scale <= 0 or scale >= len(dog_pyramid[octave]) - 1):
            return None, None, None
        # Get adjacent scale images (scale-1, scale, scale+1)
        prev_img = dog_pyramid[octave][scale - 1]
        curr_img = dog_pyramid[octave][scale]
        next_img = dog_pyramid[octave][scale + 1]

        # First derivatives (gradient vector)
        dx = (curr_img[y, x + 1] - curr_img[y, x - 1]) * 0.5
        dy = (curr_img[y + 1, x] - curr_img[y - 1, x]) * 0.5
        ds = (next_img[y, x] - prev_img[y, x]) * 0.5
        gradient = np.array([dx, dy, ds])

        # Second derivatives (Hessian matrix)
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
        #msh fahma ma3na daaaa
        # Solve for offset
        try:
            offset = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if np.any(np.abs(offset) > 1.5):
                return None, None, None
        except np.linalg.LinAlgError:
            return None, None, None  # singular Hessian, discard keypoint

        # Interpolated contrast
        contrast = curr_img[y, x] + 0.5 * np.dot(gradient, offset)
        if abs(contrast) < 0.03:
            return None, None, None  # too low contrast

        # Return refined keypoint position (floating point offset added to original)
        refined_x = x + offset[0]
        refined_y = y + offset[1]
        refined_scale = scale + offset[2]

        return refined_x, refined_y, refined_scale

    def is_edge_like(curr_img, x, y, edge_threshold=10):
        # Second order derivatives at current scale (2x2 spatial Hessian)
        dxx = curr_img[y, x + 1] - 2 * curr_img[y, x] + curr_img[y, x - 1]
        dyy = curr_img[y + 1, x] - 2 * curr_img[y, x] + curr_img[y - 1, x]
        dxy = ((curr_img[y + 1, x + 1] - curr_img[y + 1, x - 1]) -
            (curr_img[y - 1, x + 1] - curr_img[y - 1, x - 1])) * 0.25

        trace = dxx + dyy
        det = dxx * dyy - dxy ** 2

        if det <= 0:
            return True  # not stable, discard

        ratio = (trace ** 2) / det
        return ratio >= ((edge_threshold + 1) ** 2) / edge_threshold







    



                


