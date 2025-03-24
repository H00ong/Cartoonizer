import cv2
import numpy as np
# Load the image
img = cv2.imread('Image/sky.jpg')
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply median blur to reduce noise
gray = cv2.medianBlur(gray, 5)
# Detect edges using adaptive thresholding
edges = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
# Convert the image to color
color = cv2.bilateralFilter(img, 5, 200, 300)
# Combine the color image with the edges mask
cartoon = cv2.bitwise_and(color, color, mask=edges)

# 이미지의 원래 크기 구하기
(h, w) = cartoon.shape[:2]

# 원하는 너비 설정
new_width = 800

# 비율을 유지하며 새로운 높이 계산 (원본 높이/원본 너비 비율)
new_height = int((new_width / w) * h)

# 이미지 리사이즈
cartoon_resize = cv2.resize(cartoon, (new_width, new_height))

cv2.imshow("Cartoon", cartoon_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()