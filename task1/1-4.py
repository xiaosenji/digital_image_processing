from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np

class ColorChange:
    def __init__(self):
        self.Image = Image.open('../picture/Airplane.jpg')
        self.w, self.h = self.Image.size

    def toYUV(self):
        im = self.Image.convert('RGB')
        for i in range(self.w):
            for j in range(self.h):
                rgb = np.array(im.getpixel((i, j)))
                change = np.array([[0.299, 0.587, 0.144], [-0.418, -0.289, 0.437], [0.615, -0.515, -0.100]])
                yuv = list(map(int, np.dot(change, rgb.T)))
                im.putpixel((i, j), tuple(yuv))
        plt.imshow(im)
        plt.title('toYUV')
        plt.axis('off')
        plt.show()

    def toHIS(self):
        im = self.Image.convert('RGB')
        for i in range(self.w):
            for j in range(self.h):
                r, g, b = im.getpixel((i, j))
                o = math.acos(0.5 * ((r - g) + (r - b)) / (pow(pow((r - g), 2) + (r - b) * (g - b), 0.5) + 1))
                if b <= g:
                    H = o
                else:
                    H = 360 - o
                S = 1 - (3 / (r + g + b)) * min([r, g, b])
                I = (r + g + b) / 3
                his = [int(H), int(I), int(S)]
                im.putpixel((i, j), tuple(his))
        plt.imshow(im)
        plt.title('toHIS')
        plt.axis('off')
        plt.show()

    def toYCbCr(self):
        im = self.Image.convert('RGB')
        for i in range(self.w):
            for j in range(self.h):
                r, g, b = im.getpixel((i, j))
                y = 0.299 * r + 0.587 * g + 0.144 * b
                Cb = 0.568 * (b - y) + 128
                Cr = 0.713 * (r - y) + 128
                YCrCb = [int(y), int(Cr), int(Cb)]
                im.putpixel((i, j), tuple(YCrCb))
        plt.imshow(im)
        plt.title('toYCrCb')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    c = ColorChange()
    c.toYUV()
    c.toHIS()
    c.toYCbCr()
