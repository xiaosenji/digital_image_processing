import matplotlib.pyplot as plt
import numpy as np
import cv2
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

class Canny:
    def __init__(self):
        self.image = cv2.imread('../image/pirate.tif', cv2.IMREAD_GRAYSCALE)
        self.h, self.w = self.image.shape[:2]

    def sobel(self):
        # Sobel算子
        img = np.zeros((self.h, self.w), np.uint8)  # 创建结果图像
        deal = np.pad(self.image, (1, 1), mode='constant', constant_values=0)  # 填充输入图像
        robert_x = np.array([[-1, 0], [0, 1]])
        robert_y = np.array([[0, -1], [1, 0]])
        for i in range(self.h):
            for j in range(self.w):
                x = deal[i:i + 2, j:j + 2]
                y1 = np.abs(np.sum(robert_x * x))
                y2 = np.abs(np.sum(robert_y * x))
                img[i, j] = int(pow((y1 ** 2 + y2 ** 2), 0.5))
        return img

    def prewitt(self):
        # Prewitt算子
        img = np.zeros((self.h, self.w), np.uint8)
        deal = np.pad(self.image, (1, 1), 'constant', constant_values=0)
        prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        for i in range(self.h):
            for j in range(self.w):
                x = deal[i:i + 3, j:j + 3]
                y1 = np.abs(np.sum(prewitt_x * x))
                y2 = np.abs(np.sum(prewitt_y * x))
                img[i, j] = int(pow((y1 ** 2 + y2 ** 2), 0.5))
        return img

    def robert(self):
        # Roberts算子
        img = np.zeros((self.h, self.w), np.uint8)
        deal = np.pad(self.image, (1, 1), 'constant', constant_values=0)
        sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        for i in range(self.h):
            for j in range(self.w):
                x = deal[i:i + 3, j:j + 3]
                y1 = np.abs(np.sum(sobel_x * x))
                y2 = np.abs(np.sum(sobel_y * x))
                img[i, j] = int(pow((y1 ** 2 + y2 ** 2), 0.5))
        return img

    def loG(self):
        # LoG算子
        laplace = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        padding = np.zeros((self.h + 2, self.w + 2), np.uint8)
        padding[1:-1, 1:-1] = self.image
        img = np.zeros((self.h, self.w), np.uint8)
        for i in range(self.h):
            for j in range(self.w):
                window = padding[i:i + 3, j:j + 3]
                img[i, j] = np.abs(np.sum(laplace * window))
        sigma = 3
        edge = int((sigma - 1) / 2)
        mask = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
        result = np.zeros((self.h, self.w), dtype="uint8")
        for i in range(edge, self.h - edge):
            for j in range(edge, self.w - edge):
                x = img[i - edge:i + edge + 1, j - edge:j + edge + 1]
                s = 0
                for h in range(3):
                    for k in range(3):
                        s += x[h][k] * mask[h][k]
                result[i, j] = int(s)
        return result


if __name__ == '__main__':
    c = Canny()
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.title('original')
    plt.axis('off')
    plt.imshow(c.image, cmap='gray')
    plt.subplot(2, 3, 2)
    plt.title('Sobel')
    plt.axis('off')
    plt.imshow(c.sobel(), cmap='gray')
    plt.subplot(2, 3, 3)
    plt.title('Prewitt')
    plt.axis('off')
    plt.imshow(c.prewitt(), cmap='gray')
    plt.subplot(2, 3, 4)
    plt.title('Roberts')
    plt.axis('off')
    plt.imshow(c.robert(), cmap='gray')
    plt.subplot(2, 3, 5)
    plt.title('LoG')
    plt.axis('off')
    plt.imshow(c.loG(), cmap='gray')
    plt.show()
