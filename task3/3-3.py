import numpy as np
import matplotlib.pyplot as plt
import cv2


class Morphology:
    def __init__(self):
        self.image = cv2.imread('../image/lake.tif', cv2.IMREAD_GRAYSCALE)

    def deal(self):
        img1 = self.image
        opening = self.erode(img1)
        opening = self.dilate(opening)
        img2 = self.image
        closing = self.dilate(img2)
        closing = self.erode(closing)
        self.draw(opening, closing)

    def dilate(self, img):
        h, w = img.shape[:2]
        time = 1
        MF = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        result = img.copy()
        for i in range(time):
            tmp = np.pad(result, (1, 1), 'edge')
            for i in range(1, h):
                for j in range(1, w):
                    if np.sum(MF * tmp[i-1:i+2, j-1:j+2]) >= 255:
                        result[i, j] = 255
        return result

    def erode(self, img):
        h, w = img.shape[:2]
        result = img.copy()
        time = 1
        MF = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        for i in range(time):
            tmp = np.pad(result, (1, 1), 'edge')
            for i in range(1, h):
                for j in range(1, w):
                    if np.sum(MF * tmp[i-1:i+2, j-1:j+2]) < 255*4:
                        result[i, j] = 0
        return result

    def draw(self, opening, closing):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(self.image, cmap='gray')
        plt.title('original')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(opening, cmap='gray')
        plt.title('opening operation')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(closing, cmap='gray')
        plt.title('closing operation')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    m = Morphology()
    m.deal()
