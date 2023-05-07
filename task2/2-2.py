import cv2
import matplotlib.pyplot as plt
import numpy as np


class EqualizationHistogram:
    def __init__(self):
        self.Image = cv2.imread('../image/pirate.tif', cv2.IMREAD_GRAYSCALE)

    def imhist(self):
        im, L = self.Image, 256
        plt.figure(figsize=(8, 7))
        plt.subplot(2, 2, 1)
        plt.imshow(im, cmap='gray')
        plt.axis('off')
        plt.title('original')
        h, w = im.shape[:2]
        result = np.zeros((h, w))
        hist1 = dict((i, 0) for i in range(L))
        # 计算原始图像灰度级像素个数ni
        for x in range(0, h):
            for y in range(0, w):
                i = im[x, y]
                if i in hist1:
                    hist1[i] += 1
                else:
                    hist1[i] = 1
        key = [i for i in range(L)]  # key为灰度级，即0~255
        value1 = [hist1[i] for i in range(L)]  # value1为各灰度级的像素个数ni
        n = w * h
        # 计算原始图像的直方图，value2为原始图像的直方图P(i) = ni/n = ni/(w*h)
        value2 = [value1[i] / n for i in range(L)]
        plt.subplot(2, 2, 2)
        plt.bar(key, value2)
        plt.title('original histogram')
        # 计算累积直方图，value3为累积直方图Pj
        value3 = [0 for i in range(L)]
        value3[0] = value2[0]
        for j in range(1, L):
            value3[j] = value3[j - 1] + value2[j]
        # 计算灰度值变换，value4为由r变换为s后的灰度值
        value4 = [0 for i in range(L)]
        for j in range(L):
            value4[j] = int((L - 1) * value3[j] + 0.5)
        # 计算变换后各灰度级的像素个数nj，value5即为nj
        value5 = [0 for j in range(L)]
        for x in range(0, h):
            for y in range(0, w):
                i = im[x, y]
                t = value4[i]
                result[x, y] = t
                for k in range(L):
                    if t == k:
                        value5[k] = value5[k] + 1
        # 计算变换后图像的直方图，value6即为P(j)
        value6 = [value5[i] / n for i in range(L)]
        # 均衡化后的图像
        plt.subplot(2, 2, 3)
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.title('equalized')
        plt.subplot(2, 2, 4)
        plt.bar(key, value6)
        plt.title('equalized histogram')
        plt.show()


if __name__ == '__main__':
    eh = EqualizationHistogram()
    eh.imhist()
