#Image Arithmetic
import cv2
import numpy as np

img = cv2.imread('msd.jpeg')

cv2.imshow('Original', img)
cv2.waitKey(0)
M = np.ones(img.shape, dtype="uint8") * 150
# M1 = np.zeros(img.shape, dtype="uint8") + 150


added = cv2.add(img, M)
cv2.imshow('Added', added)

subtracted = cv2.subtract(img, M)
cv2.imshow('Subtract', subtracted)

mul = cv2.multiply(img,M)
cv2.imshow('MUl',mul)

cv2.waitKey(0)
cv2.destroyAllWindows()