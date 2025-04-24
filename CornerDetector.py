import cv2
import numpy as np


class CornerDetector:
    def __init__(self, image,color_image, window_size=5, threshold=0.01):
        self.image = image
        self.color_image = color_image
        self.threshold = threshold
        self.window_size = window_size
        self.corners = None
        
    #get Ix, Iy, IxIy
    def _sobel_gradients(self):
        sobelx = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=5)
        return sobelx**2, sobely**2, sobelx * sobely

    #get Gx, Gy, GxGy
    def _apply_gaussian_window(self, Ix_squared, Iy_squared, IxIy):
        Gx = cv2.GaussianBlur(Ix_squared, (self.window_size, self.window_size), 0)
        Gy = cv2.GaussianBlur(Iy_squared, (self.window_size, self.window_size), 0)
        GxGy = cv2.GaussianBlur(IxIy, (self.window_size, self.window_size), 0)
        return Gx, Gy, GxGy
    
    #harris matrix and response function 
    def compute_harris_response(self, k=0.04):
        Ix2, Iy2, Ixy = self._sobel_gradients()
        Gx, Gy, Gxy = self._apply_gaussian_window(Ix2, Iy2, Ixy)

        det_M = Gx * Gy - Gxy**2
        trace_M = Gx + Gy
        R = det_M - k * (trace_M**2)

        self.corners = R

    #Thresholding and non-maximum suppression
    def threshold_and_nms(self):    
        R = self.corners
        threshold = self.threshold * R.max()
        corner_map = np.zeros_like(R)
        corner_map[R > threshold] = 255

        # Non-maximum suppression using dilation
        dilated = cv2.dilate(R, None)   #find local maxima in R by comparing each value by it neighbors
        nms_corners = np.zeros_like(R)
        nms_corners[(R == dilated) & (R > threshold)] = 255 ## 3. Keep only those pixels that are:
                                                    #    - Equal to the dilated value (i.e. local maxima)
                                                    #    - AND above a threshold

        nms_corners = nms_corners.astype(np.uint8)
        return nms_corners
        
    def draw_corners_on_image(self, nms_corners, radius=5, color=(0, 0, 255)):
        result_img = self.color_image.copy()
        coords = np.argwhere(nms_corners)
        for y, x in coords:
            cv2.circle(result_img, (x, y), radius, color, thickness=-1)

        return result_img #to be shown in the output widget
    
    
    def compute_lambda_min_response(self):
        Ix2, Iy2, Ixy = self._sobel_gradients()
        Gx, Gy, Gxy = self._apply_gaussian_window(Ix2, Iy2, Ixy)

        # Compute the eigenvalues of the structure tensor for each pixel
        trace = Gx + Gy
        sqrt_term = np.sqrt((Gx - Gy) ** 2 + 4 * Gxy ** 2)

        lambda1 = 0.5 * (trace + sqrt_term)
        lambda2 = 0.5 * (trace - sqrt_term)

        self.corners = np.minimum(lambda1, lambda2) 
    def apply_corner_detection(self, method="harris"):
        if method == "harris":
            self.compute_harris_response()
        elif method == "lambda_min":
            self.compute_lambda_min_response()
        else:
            raise ValueError("Method must be either 'harris' or 'lambda_min'")

        nms_corners = self.threshold_and_nms()
        result_img = self.draw_corners_on_image(nms_corners)
        return result_img
