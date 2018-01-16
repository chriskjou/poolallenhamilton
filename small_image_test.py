import cv2

xcoord = int(14.223692736437574 * 23)
ycoord = int(1.2713831816794794 * 23)
interesting_count = 0
image = cv2.imread("threshold2.jpg")
test_image = image[60:850,40:1620]
cv2.imwrite("test%d.jpg" % interesting_count, test_image)
small_image = image[max(ycoord, 0): min(ycoord + 48, 1480), max(xcoord, 0): min(xcoord + 48, 1480)]
cv2.imwrite("interesting%d.jpg" % interesting_count, small_image)