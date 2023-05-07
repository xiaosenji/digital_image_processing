import cv2
import matplotlib.pyplot as plt
import numpy as np

# 二值化图像
img = cv2.imread('../image/woman_blonde.tif', cv2.IMREAD_GRAYSCALE)
h, w = img.shape[:2]
plt.figure()
plt.subplot(1, 3, 1)
plt.title('original')
plt.axis('off')
plt.imshow(img, cmap='gray')
# 全局阈值
thresh = 127
im1 = np.zeros((h, w))
for i in range(h):
    for j in range(w):
        if img[i, j] > thresh:
            im1[i, j] = 255
        else:
            im1[i, j] = 0
plt.subplot(1, 3, 2)
plt.title('global threshold')
plt.imshow(im1, cmap='gray')
plt.axis('off')
# 局部阈值
im2 = np.zeros((h, w))
im = np.pad(img, (1, 1), 'constant')
for i in range(h):
    for j in range(w):
        x = np.mean(im[i:i+3, j:j+3])
        if img[i, j] >= x:
            im2[i, j] = 255
        else:
            im2[i, j] = 0
plt.subplot(1, 3, 3)
plt.title('local threshold')
plt.imshow(im2, cmap='gray')
plt.axis('off')
plt.show()
