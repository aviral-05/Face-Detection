#image Cropping
import cv2
import numpy as pd

img = cv2.imread('msd.jpeg')

height, width = img.shape[:2]

#Let's get the starting pixel coordinates(top left, of cropping rectangle)

start_row, start_col = int(height*.25), int(width*.25)

#Let's get the ending pixel coordinates (bottom right)

end_row, end_col = int(height*.75), int(width*.75)

#Simply use the indexing to crop the image

cropped = img[start_row:end_row, start_col:end_col]

cv2.imshow('Original', img)
cv2.waitKey(0)

cv2.imshow('Cropped', cropped)
cv2.waitKey(0)

cv2.destroyAllWindows()