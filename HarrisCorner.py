import cv2
import numpy as np

class HarrisCorner:
    def __init__(self, image):
        self.image = image
        self.corners = None

    def _sobel_gradients(self):
        sobelx = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=5)
        return sobelx**2, sobely**2, sobelx * sobely

    def _apply_gaussian_window(self, Ix_squared, Iy_squared, IxIy):
        Gx = cv2.GaussianBlur(Ix_squared, (5, 5), 0)
        Gy = cv2.GaussianBlur(Iy_squared, (5, 5), 0)
        GxGy = cv2.GaussianBlur(IxIy, (5, 5), 0)
        return Gx, Gy, GxGy

    def compute_harris_response(self, k=0.04):
        Ix2, Iy2, Ixy = self._sobel_gradients()
        Gx, Gy, Gxy = self._apply_gaussian_window(Ix2, Iy2, Ixy)

        det_M = Gx * Gy - Gxy**2
        trace_M = Gx + Gy
        R = det_M - k * (trace_M**2)

        self.corners = R
        return R

    #Thresholding and non-maximum suppression
    def threshold_and_nms(self, threshold_ratio=0.001):
    
        R = self.corners
        threshold = threshold_ratio * R.max()
        corner_map = np.zeros_like(R)
        corner_map[R > threshold] = 255

        # Non-maximum suppression using dilation
        dilated = cv2.dilate(R, None)
        nms_corners = np.zeros_like(R)
        nms_corners[(R == dilated) & (R > threshold)] = 255

        self.corners = nms_corners.astype(np.uint8)
        
    def draw_corners_on_image(self, radius=7, color=(0, 0, 255)):
        result_img = self.image.copy()
        coords = np.argwhere(self.corners)
        for y, x in coords:
            cv2.circle(result_img, (x, y), radius, color, thickness=-1)

        return result_img #to be shown in the output widget