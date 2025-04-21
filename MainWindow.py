from PyQt5.QtWidgets import QMainWindow,QComboBox,QTabWidget, QSpinBox, QWidget, QApplication, QPushButton, QLabel, QSlider,QProgressBar,QGraphicsView,QGraphicsScene
from PyQt5.QtGui import QIcon
import os
import sys
import cv2
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
from CornerDetector import CornerDetector
from ImageViewer import ImageViewer

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
        detector = CornerDetector(image, color_image, window_size=window_size, threshold=threshold)
        result_img = detector.apply_corner_detection(method="harris")

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.showMaximized()
    sys.exit(app.exec_())        
    