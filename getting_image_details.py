import cv2

img = cv2.imread('test.jpeg',0)

print(img)
cv2.imshow('image',img)
k = cv2.waitKey(5000) & 0xFF

if k == 27:
	cv2.destroyAllWindows()

elif k == ord('s'):
	cv2.imwrite('test.png',img)

else:
	print("you didn't pressed any key ")
