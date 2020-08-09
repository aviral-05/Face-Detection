# import OpenCV
import cv2
 
# Load trained cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
# Read the given image
color_image = cv2.imread('msd.jpeg')
 
# Convert color image into grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
 
# Detect faces ROI
#Syntax: Classifier.detectMultiScale(input image, Scale Factor , Min Neighbors)
faces = face_cascade.detectMultiScale(gray_image, 1.1, 5) 
 
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 4)
     
# Show image
cv2.imshow('Image', color_image)
 
#wait to close window
cv2.waitKey()
 
#close all windows
cv2.destroyAllWindows()