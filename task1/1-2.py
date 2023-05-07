import cv2
import matplotlib.pyplot as plt


class Operation:
    def __init__(self):
        self.image1 = cv2.imread('../picture/Lena.jpg')
        self.image2 = cv2.imread('../picture/demoWithSalt.jpg')
        self.w, self.h = self.image1.shape[0], self.image1.shape[1]

    def add_image(self):
        # 单个图像加运算
        im = self.image1
        img_add = im + im
        plt.imshow(img_add.astype('uint8'))
        plt.axis('off')
        plt.title('add')
        plt.show()

    def subtract_image(self):
        # 单个图像减运算
        im = self.image1
        img_sub = im - 100
        plt.imshow(img_sub.astype('uint8'))
        plt.axis('off')
        plt.title('subtract')
        plt.show()

    def multiply_image(self):
        # 单个图像乘运算
        im = self.image1
        img_mul = im * 3
        plt.imshow(img_mul.astype('uint8'))
        plt.axis('off')
        plt.title('multiply')
        plt.show()

    def divide_image(self):
        # 单个图像除运算
        im = self.image1
        for i in range(self.w):
            for j in range(self.h):
                x = im[i][j]
                for k in range(len(list(x))):
                    y = list(x)[k] / 2
                    if y < 0:
                        list(x)[k] = 0
                    elif y > 255:
                        list(x)[k] = 255
        plt.imshow(im.astype('uint8'))
        plt.axis('off')
        plt.title('divide')
        plt.show()

    def index_image(self):
        # 单个图像指数运算
        im = cv2.imread('../picture/lena(g).jpg', cv2.IMREAD_GRAYSCALE)
        for i in range(self.w):
            for j in range(self.h):
                tmp = im[i][j]
                tmp = int(pow(tmp + 0, 5))
                if 0 <= tmp <= 255:
                    im[i][j] = tmp
                elif tmp > 255:
                    im[i][j] = 255
                else:
                    im[i][j] = 0
        plt.imshow(im.astype('uint8'), cmap='gray')
        plt.axis('off')
        plt.title('index')
        plt.show()

    def add_two_image(self):
        # 两幅图像加运算
        im1, im2 = self.image1, self.image2
        im3 = im1 + im2
        plt.imshow(im3.astype('uint8'))
        plt.axis('off')
        plt.title('add_two')
        plt.show()

    def subtract_two_image(self):
        # 两幅图像减运算
        im1, im2 = self.image1, self.image2
        im3 = im2 - im1 * 0.5
        plt.imshow(im3.astype('uint8'))
        plt.axis('off')
        plt.title('subtract_two')
        plt.show()


if __name__ == '__main__':
    image = Operation()
    image.add_image()
    image.subtract_image()
    image.multiply_image()
    image.divide_image()
    image.index_image()
    image.add_two_image()
    image.subtract_two_image()