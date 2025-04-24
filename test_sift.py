import numpy as np
import cv2
import matplotlib.pyplot as plt
from SIFT import SIFT  # Your fixed SIFT implementation

class SIFTVisualizer:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # Initialize your SIFT
        self.sift = SIFT(self.image)
        
    def visualize_pyramids(self):
        """Visualize Gaussian and DoG pyramids"""
        base_image = self.sift.generate_base_image(self.sift.image)
        gaussian_pyramid = self.sift.generate_gaussian_images(base_image)
        dog_pyramid = self.sift.generate_dog_images()
        
        # Visualize first octave
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle("Gaussian Pyramid (First Octave)", fontsize=16)
        for i, img in enumerate(gaussian_pyramid[0][:5]):
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f"Layer {i}")
            axes[0, i].axis('off')
        
        # Visualize DoG of first octave
        for i, img in enumerate(dog_pyramid[0][:4]):
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title(f"DoG {i+1}-{i}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_keypoints(self):
        """Visualize detected keypoints with proper coordinate scaling"""
        keypoints = self.sift.find_scale_space_extrema()
        
        # Create visualization
        img_display = self.image.copy()
        
        for kp in keypoints:
            x, y, scale, octave = kp
            x, y = int(x), int(y)  # Convert to integer coordinates
            
            # Calculate keypoint size based on scale and octave
            size = int(1.6 * (2 ** (scale / 3)) * (2 ** octave))
            
            # Only draw if within image bounds
            if 0 <= x < img_display.shape[1] and 0 <= y < img_display.shape[0]:
                cv2.circle(img_display, (x, y), size, (0, 255, 0), 1)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        plt.title(f"Your SIFT Keypoints ({len(keypoints)} detected)")
        plt.axis('off')
        plt.show()
    def run_visualizations(self):
        """Run all visualization steps"""
        print("Visualizing Gaussian and DoG pyramids...")
        self.visualize_pyramids()
        
        print("\nVisualizing detected keypoints...")
        self.visualize_keypoints()

if __name__ == "__main__":
    image_path = r"C:\Users\joody\Downloads\cat-apple-8883021.jpg"
    try:
        visualizer = SIFTVisualizer(image_path)
        visualizer.run_visualizations()
    except Exception as e:
        print(f"Error: {str(e)}")