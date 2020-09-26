from utils.anpr import ANPR
from imutils import paths
import argparse
import imutils
import cv2

def cleanup_text(text):
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-d", "--debug", type=int, default=-1,
	help="whether or not to show additional visualizations")
args = vars(ap.parse_args())


anpr = ANPR(debug=args["debug"] > 0)
imagePaths = sorted(list(paths.list_images(args["input"])))

for imagePath in imagePaths:

	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	
	(lpText, bbox) = anpr.find_and_ocr(image)
	
	if lpText is not None and bbox is not None:
		
		(x, y, w, h) = bbox
		cv2.rectangle(image, (x ,y), (x+w ,y+h ), (255, 0, 0), 3)
		cv2.putText(image, cleanup_text(lpText), (x, y - 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
		
		print("[INFO] {}".format(lpText))
		cv2.imshow("Output ANPR", image)
		cv2.imwrite('Output/'+imagePath.split("/")[-1],image)
		cv2.waitKey(0)	