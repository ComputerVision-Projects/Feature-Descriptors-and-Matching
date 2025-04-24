import numpy as np
from PIL import Image, ImageDraw
import time

class SSDMatcher:
    def __init__(self):
        self.min_val = float('inf')
        self.min_loc = (0, 0)
        self.last_match_time = 0
    def sum_squared_differences(self, roi, target):
        """Compute sum of squared differences between ROI and target"""
        return np.sum((roi - target)**2)
    
    def match_template(self, image, template):
        """
        Match template using SSD
        Args:
            image: PIL Image (main image)
            template: PIL Image (template to search for)
        Returns:
            (x, y) coordinates of best match
        """
        # Convert images to numpy arrays and grayscale
        img = np.array(image.convert('L'), dtype=np.float32)
        target = np.array(template.convert('L'), dtype=np.float32)
        
        h, w = img.shape
        th, tw = target.shape
        
        self.min_val = float('inf')
        self.min_loc = (0, 0)
        # Start timing
        start_time = time.time()
        # Slide template across the image
        for y in range(h - th):
            for x in range(w - tw):
                roi = img[y:y+th, x:x+tw]
                score = self.sum_squared_differences(roi, target)
                
                if score < self.min_val:
                    self.min_val = score
                    self.min_loc = (x, y)
        # Store the elapsed time
        self.last_match_time = time.time() - start_time
        return self.min_loc
    
    def draw_result(self, image, template):
        """Draw rectangle around the matched area"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        x, y = self.match_template(image, template)
        w, h = template.size
        draw.rectangle([x, y, x+w, y+h], outline="blue", width=2)
        return img_copy
    
    def get_last_match_time_SSD(self):
        """Returns the time taken for the last match operation in seconds"""
        return self.last_match_time