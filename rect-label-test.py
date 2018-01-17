import cv2
import pandas as pd
from PIL import Image
import glob
print(cv2.__version__)

vids = glob.glob('rect/*.jpg')
def draw_boxes(image_name):
    # selected_value = full_labels[full_labels.filename == image_name]
    img = cv2.imread('{}'.format(image_name))
    xmin = 40
    xmax = 80
    ymin = 40
    ymax = 80
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    return img

print(Image.fromarray(draw_boxes("threshold.jpg")))