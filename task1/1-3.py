import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


class Transform:
    def __init__(self):
        self.Image = cv2.imread('../picture/High-Resolution-3-ALKVC.jpg')

    def translate(self):
        # 图像平移
        im = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        shift = np.zeros(shape=(h, w, 3), dtype=int)
        M = np.array([[1, 0, 100], [0, 1, 100], [0, 0, 1]])
        for i in range(h):
            for j in range(w):
                rgb, xy = im[i][j], np.array([i, j, 1])
                change = np.dot(M, xy.T)
                u, v = int(change[0]), int(change[1])
                if u < shift.shape[0] and v < shift.shape[1]:
                    shift[u][v] = rgb
        plt.imshow(shift)
        plt.title('translate')
        plt.show()

    def magnify(self):
        # 图像放大
        im = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        scrW, scrH = self.Image.shape[:2]
        large_img = np.zeros((scrW * 2, scrH * 2, 3), dtype=np.uint8)
        for i in range(scrW * 2 - 1):
            for j in range(scrH * 2 - 1):
                scrx = round(i * (scrW / (scrW * 2)))
                scry = round(j * (scrH / (scrH * 2)))
                large_img[i, j] = im[scrx, scry]
        plt.imshow(large_img)
        plt.title('magnify')
        plt.show()

    def shrink(self):
        # 图像缩小
        im = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        scrW, scrH = self.Image.shape[:2]
        small_img = np.zeros((int(scrW * 0.5), int(scrH * 0.5), 3), dtype=np.uint8)
        for i in range(int(scrW * 0.5) - 1):
            for j in range(int(scrH * 0.5) - 1):
                scrx = round(i * (scrW / (scrW * 0.5)))
                scry = round(j * (scrH / (scrH * 0.5)))
                small_img[i, j] = im[scrx, scry]
        plt.imshow(small_img)
        plt.title('shrink')
        plt.show()

    def rotate(self):
        # 旋转角度
        im = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        height, width = self.Image.shape[:2]
        angle = 30
        if int(angle / 90) % 2 == 0:
            reshape_angle = angle % 90
        else:
            reshape_angle = 90 - (angle % 90)
        reshape_radian = math.radians(reshape_angle)  # 角度转弧度
        # 三角函数计算出来的结果会有小数，所以做了向上取整的操作。
        new_height = math.ceil(height * np.cos(reshape_radian) + width * np.sin(reshape_radian))
        new_width = math.ceil(width * np.cos(reshape_radian) + height * np.sin(reshape_radian))
        rotation_img = np.zeros(shape=(new_height, new_width, 3), dtype=np.uint8)
        radian = math.radians(angle)
        cos_radian, sin_radian = np.cos(radian), np.sin(radian)
        dx = 0.5 * new_width + 0.5 * height * sin_radian - 0.5 * width * cos_radian
        dy = 0.5 * new_height - 0.5 * width * sin_radian - 0.5 * height * cos_radian
        for y0 in range(height):
            for x0 in range(width):
                x = x0 * cos_radian - y0 * sin_radian + dx
                y = x0 * sin_radian + y0 * cos_radian + dy
                rotation_img[int(y) - 1, int(x) - 1] = im[int(y0), int(x0)]  # 因为整体映射的结果会比偏移一个单位，所以这里x,y做减一操作。
        plt.imshow(rotation_img)
        plt.title('rotate')
        plt.show()


if __name__ == "__main__":
    t = Transform()
    t.translate()
    t.magnify()
    t.shrink()
    t.rotate()
