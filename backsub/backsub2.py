import numpy as np
import cv2


original_image = cv2.imread('../threshold.jpg', cv2.IMREAD_COLOR)
gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
background_image = cv2.imread('../threshold2.jpg', cv2.IMREAD_COLOR)
gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)

# foreground = np.absolute(gray_original - gray_background)
# foreground = cv2.absdiff(gray_original, gray_background)
foreground = cv2.absdiff(original_image, background_image)
# foreground[foreground > 0] = 255

cv2.imshow('Original Image', foreground)
cv2.waitKey(0)