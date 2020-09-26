from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2
from utils.PlateDetection import PlateDetector

class ANPR:

    def __init__(self, debug=False):
        self.debug = debug
        self.plateDetector = PlateDetector(type_of_plate='RECT_PLATE', minPlateArea=500, maxPlateArea=10000)

    def build_tesseract_options(self, psm=7):
        # tell Tesseract to only OCR alphanumeric characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        options += " --psm {}".format(psm)
        return options

    def find_and_ocr(self, image, psm=7):
        lpText = None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lpCnt, boxes = self.plateDetector.find_possible_plates(image)
        try:
            x,y,w,h = boxes[0]
            lp = image[y:y+h,x:x+w]
            # OCR the license plate
            options = self.build_tesseract_options(psm=psm)
            lpText = pytesseract.image_to_string(lp, config=options)
            if self.debug:
                cv2.imshow('License Plate', lp)
                if waitKey:
                    cv2.waitKey(0)
            return lpText, [x,y,w,h]    
        except Exception as e:
            print(e)
            return None, None