from PyQt5.QtWidgets import QFileDialog, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QEvent, Qt
import cv2
import numpy as np

class ImageViewer(QWidget):
    def __init__(self, input_view=None, output_view=None, index=0, img_num=None,mode=False, widget=0):
        super().__init__()
        self.img_num = img_num
        self.index = index
        self._image_path = None
        self.img_data = None
        self._is_grey = False
        self.input_view = input_view
        self.output_view = output_view

        self.mode = mode #to display colored image
        self.widget = widget
        self.label=None
        self.setup_double_click_event()
        if self.widget == 2 and self.input_view:
            print("enter")
            self.input_view.setMouseTracking(True)
            self.input_view.installEventFilter(self)
       

       
    
    def setup_double_click_event(self):
        if self.input_view:
            self.input_view.mouseDoubleClickEvent = self.handle_double_click
    
    

    def handle_double_click(self, event=None):
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if file_path:
            self.browse_image(file_path)
    
    def browse_image(self, image_path):
        self._image_path = image_path
        if self.check_extension():
         if self.mode:
                print("enter333")
                self.img_data = cv2.imread(self._image_path, cv2.IMREAD_COLOR)
                self._is_grey = False
         else:   

            if self.index in [0, 2]:
                self.img_data = cv2.imread(self._image_path, cv2.IMREAD_GRAYSCALE)
                self._is_grey = True
            else:
                self.img_data = cv2.imread(self._image_path, cv2.IMREAD_COLOR)
                self._is_grey = False

         if self.img_data is None:
                print("Error loading image.")
                return

        if self.output_view:
             for child in self.output_view.findChildren(QLabel):
              child.deleteLater() 

        self._processed_image = self.img_data  # Store the processed image


        if self.input_view:
                if self.index==0:
                        self.display_image(self.img_data, self.input_view)
                elif self.index==2:
                    self.display_image(self.img_data, self.input_view)

            
    
    def display_output_image(self, processed_img=None, output=None):
        if processed_img is None:
            processed_img = self._processed_image  
        if processed_img is None:
            print("No processed image to display.")
            return
        if output:
            self.display_image(processed_img, output)
        elif self.output_view:
            self.display_image(processed_img, self.output_view)
    
    
    def display_image(self, img, target):
        
        if img is None or not isinstance(img, np.ndarray):
            print("Invalid image data.")
            return

        
         # Determine image mode and convert accordingly
        if self.mode:  # Color mode
         print ("enter2")
         if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)

         else:
            print("Expected a color image but got grayscale.")
            return
        else:  # Grayscale mode
         if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         q_image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
       

       
        self.scale_x = img.shape[1] / target.width()
        self.scale_y = img.shape[0] / target.height()
        if q_image.isNull():
            print("Failed to create QImage.")
            return

        pixmap = QPixmap.fromImage(q_image)

        
        for child in target.findChildren(QLabel):
            child.deleteLater()

        self.label = QLabel(target)
        self.label.setPixmap(pixmap.scaled(target.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        self.label.setScaledContents(True)
        self.label.setGeometry(0, 0, target.width(), target.height())

        # Ensure the label is visible and on top
        self.label.show()
        self.label.raise_()

        print(f"{'Color' if self.mode else 'Grayscale'} image displayed in widget with size: {target.size() , self.input_view }")



    def check_extension(self):
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        if not any(self._image_path.lower().endswith(ext) for ext in valid_extensions):
            print("Invalid image file extension.")
            self._image_path = None  
            return False
        print("Valid image file extension.")
        return True
    


    def check_grey_scale(self, image):
        if len(image.shape) == 2:
            return True  
        b, g, r = cv2.split(image)
        return cv2.countNonZero(b - g) == 0 and cv2.countNonZero(b - r) == 0
    
    def get_loaded_image(self):
        print("Returning loaded image:", type(self.img_data), self.img_data.shape if self.img_data is not None else "None")

        return self.img_data


