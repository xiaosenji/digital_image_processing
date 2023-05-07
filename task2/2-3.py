import matplotlib.pyplot as plt
import numpy as np
import cv2
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)


class ImageEnhancement:
    def __init__(self):
        self.Image = cv2.imread('../image/pirate.tif', cv2.IMREAD_GRAYSCALE)
        self.h, self.w = self.Image.shape[:2]

    def deal(self):
        self.global_gray_linear_transformation()
        self.gamma_transformation()
        self.image_sharpen()

    def global_gray_linear_transformation(self):
        # 全局灰度线性变换
        img = self.Image
        list0, maxV, minV = self.histogram(self.Image)
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(img, vmin=0, vmax=255, cmap='gray')
        plt.axis('off')
        plt.title('original')
        plt.subplot(2, 2, 2)
        plt.plot(list0)
        im = np.zeros((self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                x = img[i, j]
                y = ((x - minV) * 255) / (maxV - minV)
                im[i, j] = int(y)
        plt.subplot(2, 2, 3)
        plt.imshow(im, vmin=0, vmax=255, cmap='gray')
        plt.title('after global gray linear transform')
        plt.axis('off')
        list1, maxB, minB = self.histogram(im)  # 统计变换后图像的各灰度值像素的个数
        plt.subplot(2, 2, 4)
        plt.plot(list1)
        plt.show()

    def histogram(self, img):
        # 统计各灰度值的像素个数
        hist, maxV, minV = [0] * 256, 0, 255
        for i in range(self.h):
            for j in range(self.w):
                x = int(img[i, j])
                if x > maxV:
                    maxV = x
                if x < minV:
                    minV = x
                hist[x] += 1
        return hist, maxV, minV

    def gamma_transformation(self):
        # 伽马变换
        img = self.Image
        plt.figure()
        plt.subplot(1, 2, 1)
        # 使用matplotlib中的imshow显示图像，注意参数的含义，不加参数试试
        plt.imshow(img, vmin=0, vmax=255, cmap=plt.cm.gray)
        plt.title('original')
        plt.axis('off')
        invGamma = 1.0 / 2.2
        im = np.zeros((self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                x = img[i, j]
                y = ((x / 255.0) ** invGamma) * 255
                im[i, j] = int(y)
        plt.subplot(1, 2, 2)
        plt.imshow(im, vmin=0, vmax=255, cmap='gray')
        plt.title('after Gamma transform')
        plt.axis('off')
        plt.show()

    def image_sharpen(self):
        # 图像锐化
        img = self.Image
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.title('Origin')
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        # Roberts算子
        img1 = np.zeros((self.h, self.w), np.uint8)  # 创建结果图像
        deal1 = np.pad(img, (1, 1), mode='constant', constant_values=0)  # 填充输入图像
        robert_x = np.array([[-1, 0], [0, 1]])
        robert_y = np.array([[0, -1], [1, 0]])
        for i in range(self.h):
            for j in range(self.w):
                x = deal1[i:i + 2, j:j + 2]
                y1 = np.abs(np.sum(robert_x * x))
                y2 = np.abs(np.sum(robert_y * x))
                img1[i, j] = int(pow((y1 ** 2 + y2 ** 2), 0.5))
        img1 = img1 + img
        plt.subplot(2, 3, 2)
        plt.title('Roberts')
        plt.axis('off')
        plt.imshow(img1, cmap='gray')
        # Prewitt算子
        img2 = np.zeros((self.h, self.w), np.uint8)
        deal2 = np.pad(img, (1, 1), 'constant', constant_values=0)
        prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        for i in range(self.h):
            for j in range(self.w):
                x = deal2[i:i + 3, j:j + 3]
                y1 = np.abs(np.sum(prewitt_x * x))
                y2 = np.abs(np.sum(prewitt_y * x))
                img2[i, j] = int(pow((y1 ** 2 + y2 ** 2), 0.5))
        img2 = img2 + img
        plt.subplot(2, 3, 3)
        plt.title('Prewitt')
        plt.axis('off')
        plt.imshow(img2, cmap='gray')
        # Sobel算子
        img3 = np.zeros((self.h, self.w), np.uint8)
        deal3 = np.pad(img, (1, 1), 'constant', constant_values=0)
        sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        for i in range(self.h):
            for j in range(self.w):
                x = deal3[i:i + 3, j:j + 3]
                y1 = np.abs(np.sum(sobel_x * x))
                y2 = np.abs(np.sum(sobel_y * x))
                img3[i, j] = int(pow((y1 ** 2 + y2 ** 2), 0.5))
        img3 = img3 + img
        plt.subplot(2, 3, 4)
        plt.title('Sobel')
        plt.axis('off')
        plt.imshow(img3, cmap='gray')
        # Laplacian算子
        laplace = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        padding = np.zeros((self.h + 2, self.w + 2), np.uint8)  # 图像周边填充0
        padding[1:-1, 1:-1] = img
        img4 = np.zeros((self.h, self.w), np.uint8)  # 创建结果图像
        # 卷积运算
        for i in range(self.h):  # 5*5的矩阵从左到右运算3次，从上到下运算3次
            for j in range(self.w):
                window = padding[i:i + 3, j:j + 3]
                img4[i, j] = np.abs(np.sum(laplace * window))  # 矩阵内积
        img4 = img4 + img
        plt.subplot(2, 3, 5)
        plt.title('Laplacian')
        plt.axis('off')
        plt.imshow(img4, cmap='gray')
        plt.show()


if __name__ == '__main__':
    ie = ImageEnhancement()
    ie.deal()
