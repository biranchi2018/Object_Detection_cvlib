'''

Object Detection with Less Than 10 Lines of Code Using Python :
================================================================

https://towardsdatascience.com/object-detection-with-less-than-10-lines-of-code-using-python-2d28eebc5b11

https://www.cvlib.net/


Required Python Libraries :
============================
opencv-python
cvlib
matplotlib
tensorflow

'''



import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

im = cv2.imread('cat.jpeg')
bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)

plt.imshow(output_image)
plt.show()

