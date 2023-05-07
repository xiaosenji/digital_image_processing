import matplotlib.pyplot as plt
import numpy as np
import cv2


class Filter:
    def __init__(self):
        self.Image = cv2.imread('../image/walkbridge.tif', cv2.IMREAD_GRAYSCALE)

    def medium_filter(self):
        # 中值滤波
        k = 3
        imarray = self.Image
        height, width = imarray.shape[0], imarray.shape[1]
        edge = int((k - 1) / 2)
        new_arr = np.zeros((height, width), dtype="uint8")
        for i in range(edge, height - edge):
            for j in range(edge, width - edge):
                new_arr[i, j] = np.median(imarray[i - edge:i + edge + 1, j - edge:j + edge + 1])
        return new_arr

    def gauss_filter(self):
        # 高斯滤波
        img = self.Image
        height, width = self.Image.shape[:2]
        sigma = 3
        edge = int((sigma - 1) / 2)
        mask = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
        result = np.zeros((height, width), dtype="uint8")
        for i in range(edge, height - edge):
            for j in range(edge, width - edge):
                x = img[i - edge:i + edge + 1, j - edge:j + edge + 1]
                s = 0
                for h in range(3):
                    for k in range(3):
                        s += x[h][k] * mask[h][k]
                result[i, j] = int(s)
        return result

    def mean_filter(self):
        # 均值滤波
        img, k = self.Image, 3
        height, width = img.shape[:2]
        edge = int((k - 1) / 2)
        new_arr = np.zeros((height, width), dtype="uint8")
        for i in range(edge, height - edge):
            for j in range(edge, width - edge):
                new_arr[i, j] = np.mean(img[i - edge:i + edge + 1, j - edge:j + edge + 1])
        return new_arr

    def frequency_filter(self):
        D = 10
        img = self.Image
        # 傅里叶变换
        f1 = np.fft.fft2(img)
        # 使用np.fft.fftshift()函数实现平移，让直流分量输出图像的重心
        f1_shift = np.fft.fftshift(f1)
        # 实现理想低通滤波器
        rows, cols = img.shape[:2]
        crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
        mask = np.zeros((rows, cols), dtype='uint8')  # 生成rows行，从cols列的矩阵，数据格式为uint8
        # 将距离频谱中心距离小于D的低通信息部分设置为1，属于低通滤波
        for i in range(rows):
            for j in range(cols):
                if np.sqrt(i * i + j * j) <= D:
                    mask[crow - D:crow + D, ccol - D:ccol + D] = 1
        f1_shift = f1_shift * mask
        # 傅里叶逆变换
        f_ishift = np.fft.ifftshift(f1_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('original')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img_back, cmap='gray')
        plt.title('after Fourier transform')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    f = Filter()
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(f.Image, cmap='gray')
    plt.title('original')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(f.medium_filter(), cmap='gray')
    plt.title('median filter')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(f.gauss_filter(), cmap='gray')
    plt.title('Gaussian filter')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(f.mean_filter(), cmap='gray')
    plt.title('average filter')
    plt.axis('off')
    plt.show()
    f.frequency_filter()
