import numpy as np
from PIL import Image, ImageDraw
import time
class NCCMatcher:
    def __init__(self):
        self.max_val = 0
        self.max_loc = (0, 0)
        self.last_match_time = 0
    def normalized_cross_correlation(self, roi, target):
        """Compute normalized cross-correlation between ROI and target"""
        corr = np.sum(roi * target)
        norm = np.sqrt(np.sum(roi**2)) * np.sqrt(np.sum(target**2))
        return corr / (norm + 1e-10)  # Small epsilon to avoid division by zero
    
    def match_template(self, image, template):
        """
        Match template using NCC
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
        
        self.max_val = -1
        self.max_loc = (0, 0)
        # Start timing
        start_time = time.time()
        # Slide template across the image
        for y in range(h - th):
            for x in range(w - tw):
                roi = img[y:y+th, x:x+tw]
                score = self.normalized_cross_correlation(roi, target)
                
                if score > self.max_val:
                    self.max_val = score
                    self.max_loc = (x, y)
        # Store the elapsed time
        self.last_match_time = time.time() - start_time
        return self.max_loc
    
    def draw_result(self, image, template):
        """Draw rectangle around the matched area"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        x, y = self.match_template(image, template)
        w, h = template.size
        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
        return img_copy


    def get_last_match_time_NCC(self):
        """Returns the time taken for the last match operation in seconds"""
        return self.last_match_time
