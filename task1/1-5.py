import cv2
import matplotlib.pyplot as plt
import numpy as np

# 灰度图像
image = cv2.imread('../picture/Baboon.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]
im = np.zeros((h, w, 1), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        r, g, b = image[i, j]
        gray = r * 0.2989 + g * 0.587 + b * 0.114
        im[i, j] = int(gray)
plt.figure()
plt.subplot(1, 2, 1)
plt.title('original')
plt.imshow(image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('graying')
plt.imshow(im, cmap='gray')
plt.axis('off')
plt.show()
