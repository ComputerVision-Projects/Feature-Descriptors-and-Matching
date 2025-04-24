from PyQt5.QtWidgets import QMainWindow,QComboBox,QTabWidget, QSpinBox, QWidget, QApplication, QPushButton, QLabel, QSlider,QProgressBar,QCheckBox,QMessageBox
from PyQt5.QtGui import QIcon
import os
import sys
import cv2
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
from CornerDetector import CornerDetector
from ImageViewer import ImageViewer
from NCCMatcher import NCCMatcher
from SSDMatcher import SSDMatcher
from PIL import Image
import cv2
import numpy as np
import time
from SIFT import SIFT
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        loadUi("MainWindow.ui", self)

        #corner detector widgets
        self.input_image = self.findChild(QWidget, "inputImage")
        self.corners_image = self.findChild(QWidget, "cornersImage")


        self.input_viewer = ImageViewer(input_view=self.input_image, mode=True)
        self.corners_viewer = ImageViewer(output_view=self.corners_image, mode=True)

        #corner detector parameters
        self.window_slider = self.findChild(QSlider, "windowSlider")
        self.threshold_slider = self.findChild(QSlider, "thresholdSlider")

        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(50)

        self.window_slider_label = self.findChild(QLabel, "windowLabel")
        self.threshold_slider_label = self.findChild(QLabel, "thresholdLabel")

        self.window_slider.valueChanged.connect(self.update_window_label)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)


        #corner detector methods
        self.apply_harris_method = self.findChild(QPushButton, "harrisButton")
        self.apply_lambda_min_method  = self.findChild(QPushButton, "lambdaButton")

        self.apply_harris_method.clicked.connect(self.apply_harris)
        self.apply_lambda_min_method.clicked.connect(self.apply_lambda_min)
      #for tab2 NCC& SSD 
        self.input_whole_image = self.findChild(QWidget, "inputImageMatch")
        self.input_image_template = self.findChild(QWidget, "templateImage")
        self.output_image_matched = self.findChild(QWidget, "resultImage")
        self.apply_matching_method = self.findChild(QPushButton, "applyMatch")
        self.select_method = self.findChild(QComboBox, "matchingCombo")
        self.progress_bar = self.findChild(QProgressBar, "progressBar")
        self.time_label = self.findChild(QLabel, "processingTimeLabel")
        self.SIFT_check_box = self.findChild(QCheckBox, "siftCheck")





        self.input_viewer1_tab2 = ImageViewer(input_view=self.input_whole_image, mode=True)
        self.input_viewer2_tab2 = ImageViewer(input_view=self.input_image_template, mode=True)
        self.output_viewer_matched = ImageViewer(output_view=self.output_image_matched, mode=True)
        # Create matcher instances
        self.ncc_matcher = NCCMatcher()
        self.ssd_matcher = SSDMatcher()
        
        # Connect the apply button click
        self.apply_matching_method.clicked.connect(self.on_apply_clicked)

        


    def on_apply_clicked(self):
     # Get images from viewers
     self.whole_img_data = self.input_viewer1_tab2.get_loaded_image()
     self.template_img_data = self.input_viewer2_tab2.get_loaded_image()
        
     if self.SIFT_check_box.isChecked():
        self.apply_sift_matching()
     else:
        self.apply_template_matching()


    def apply_harris(self):
        image = self.input_viewer.get_loaded_image()
        if image is None:
            print("No image loaded.")
            return
        color_image = image.copy()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        window_size = self.window_slider.value()|1
        threshold = self.threshold_slider.value() / 1000.0
        # Time the detection
        start_time = time.time()
    

        detector = CornerDetector(image, color_image, window_size=window_size, threshold=threshold)
        result_img = detector.apply_corner_detection(method="harris")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Corner detection completed in {elapsed_time:.4f} seconds.")

        self.corners_viewer.display_output_image(result_img)

    def apply_lambda_min(self):
        image = self.input_viewer.get_loaded_image()
        if image is None:
            print("No image loaded.")
            return
        color_image = image.copy()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        window_size = self.window_slider.value()|1
        threshold = self.threshold_slider.value() / 1000.0
        detector = CornerDetector(image, color_image, window_size=window_size, threshold=threshold)
        result_img = detector.apply_corner_detection(method="lambda_min")

        self.corners_viewer.display_output_image(result_img)


    def update_window_label(self):
        window_size = self.window_slider.value()
        if window_size % 2 == 0:
            window_size += 1
        self.window_slider_label.setText(str(window_size))  

    def update_threshold_label(self):
        threshold = self.threshold_slider.value()/1000.0
        self.threshold_slider_label.setText(str(threshold))     


    def apply_template_matching(self):
        # Reset progress bar and time label
        self.progress_bar.setValue(0)
        self.time_label.setText("Processing...")
        # Get the selected method from combobox
        method = self.select_method.currentText()
        
        
        # Check if images are loaded
        if self.whole_img_data  is None or  self.template_img_data is None:
            QMessageBox.warning(self, "Warning", "Please load both images first!")
            return
        
        # Convert OpenCV images to PIL format (required by your matchers)
        try:
            whole_img_pil = Image.fromarray(cv2.cvtColor(self.whole_img_data , cv2.COLOR_BGR2RGB))
            template_img_pil = Image.fromarray(cv2.cvtColor( self.template_img_data, cv2.COLOR_BGR2RGB))
        except:
            # Handle grayscale images
            whole_img_pil = Image.fromarray(self.whole_img_data )
            template_img_pil = Image.fromarray( self.template_img_data)
        
         # Simulate progress
        self.progress_bar.setValue(40)
        QApplication.processEvents()
        # Perform matching based on selected method
        if "NCC" in method:
            # Get matched result
            result_img_pil = self.ncc_matcher.draw_result(whole_img_pil, template_img_pil)
            match_time = self.ncc_matcher.get_last_match_time_NCC()
        elif "SSD" in method:
           result_img_pil = self.ssd_matcher.draw_result(whole_img_pil, template_img_pil)
           match_time = self.ssd_matcher.get_last_match_time_SSD()
        else:
          QMessageBox.warning(self, "Warning", "Please select a valid matching method!")
          return
        
        # Convert PIL image back to numpy array for display
        result_img = np.array(result_img_pil)
        
        # Convert RGB to BGR if needed (OpenCV uses BGR)
        if len(result_img.shape) == 3 and result_img.shape[2] == 3:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
        # Display the result
        self.output_viewer_matched.display_output_image(result_img)
        self.progress_bar.setValue(100)
       # Set time in label
        self.time_label.setText(f"{method} completed in {match_time:.2f} seconds")



    def apply_sift_matching(self):
     image1 = self.whole_img_data
     image2 = self.template_img_data
     self.progress_bar.setValue(0)
     self.time_label.setText("Processing with SIFT...")
     QApplication.processEvents()

     start_time = time.time()

     sift1 = SIFT(image1)
     raw_kp1 = sift1.find_scale_space_extrema()
     oriented_kp1 = sift1.assign_orientation(raw_kp1)
     desc1 = sift1.compute_descriptors(oriented_kp1)
     self.progress_bar.setValue(30)
     QApplication.processEvents()
     sift2 = SIFT(image2)
     raw_kp2 = sift2.find_scale_space_extrema()
     oriented_kp2 = sift2.assign_orientation(raw_kp2)
     desc2 = sift2.compute_descriptors(oriented_kp2)
     self.progress_bar.setValue(60)
     QApplication.processEvents()
     matches = self.get_good_matches(desc1, desc2)
     self.progress_bar.setValue(80)
     QApplication.processEvents()
     print(f"Found {len(matches)} good matches")

     matched_image = self.draw_matches(image1, oriented_kp1, image2, oriented_kp2, matches)
     
    # Display matched image using the custom ImageViewer
     self.output_viewer_matched.display_output_image(matched_image)

    # Finalize
     end_time = time.time()
     elapsed_time = end_time - start_time

     self.progress_bar.setValue(100)
     self.time_label.setText(f"SIFT completed in {elapsed_time:.2f} seconds")


    def draw_matches(self,img1, kp1, img2, kp2, matches, max_matches=50):
     h1, w1 = img1.shape[:2]
     h2, w2 = img2.shape[:2]
     height = max(h1, h2)
     matched_image = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
     matched_image[:h1, :w1] = img1 if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
     y_offset = (height - h2) // 2
     matched_image[y_offset:y_offset + h2, w1:] = (
        img2 if img2.ndim == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    )
     for i, (idx1, idx2) in enumerate(matches[:max_matches]):
        x1, y1, _ = kp1[idx1]
        x2, y2, _ = kp2[idx2]

        pt1 = (int(x1), int(y1))
        pt2 = (int(x2) + w1, int(y2) + y_offset)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(matched_image, pt1, pt2, color, 1)
        cv2.circle(matched_image, pt1, 3, color, -1)
        cv2.circle(matched_image, pt2, 3, color, -1)
     return matched_image 


    def get_good_matches(self,descriptors1, descriptors2, ratio_thresh=0.5):
     descriptors2 = np.array(descriptors2)
     good_matches = []
     for i, d1 in enumerate(descriptors1):
        distances = np.sum((descriptors2 - d1) ** 2, axis=1)
        nearest = np.argsort(distances)
        if distances[nearest[0]] < ratio_thresh * distances[nearest[1]]:
            good_matches.append((i, nearest[0]))
     return good_matches


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.showMaximized()
    sys.exit(app.exec_())        
    