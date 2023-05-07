import numpy as np
import cv2
import matplotlib.pyplot as plt


class Segmentation:
    def __init__(self):
        self.image = cv2.imread('../picture/lena(g).jpg', cv2.IMREAD_GRAYSCALE)

    def segmentation(self):
        # 图像分割
        img = self.image
        height, width = img.shape[:2]
        # 求区域高斯加权值
        sigma = 3
        edge = int((sigma - 1) / 2)
        mask = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
        gauss = np.zeros((height, width), dtype="uint8")
        for i in range(edge, height - edge):
            for j in range(edge, width - edge):
                x = img[i - edge:i + edge + 1, j - edge:j + edge + 1]
                s = 0
                for h in range(3):
                    for k in range(3):
                        s += x[h][k] * mask[h][k]
                gauss[i, j] = int(s)
        # 设置超参数α∈[0,1]，当S>(1−α)T时把原像素点二值化为255
        alpha = 0.05
        result = np.uint8(img > (1 - alpha) * gauss) * 255
        plt.imshow(result, cmap='gray')
        plt.title('image segmentation')
        plt.axis('off')
        plt.show()

    def regionGrow(self):
        # 区域生长
        img = self.image
        height, weight = img.shape[:2]
        seedMark, seedList, seeds = np.zeros((height, weight)), [], [(300, 270)]
        thresh = 5
        for seed in seeds:
            seedList.append(seed)
        label = 1
        connects = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        while len(seedList) > 0:
            current_point = seedList.pop(0)
            seedMark[current_point[0], current_point[1]] = label
            for i in range(8):
                tmp_x = current_point[0] + connects[i][0]
                tmp_y = current_point[1] + connects[i][1]
                if tmp_x < 0 or tmp_y < 0 or tmp_x >= height or tmp_y >= weight:
                    continue
                grayDiff = abs(int(img[current_point[0], current_point[1]]) - int(img[tmp_x, tmp_y]))
                if grayDiff < thresh and seedMark[tmp_x, tmp_y] == 0:
                    seedMark[tmp_x, tmp_y] = label
                    seedList.append((tmp_x, tmp_y))
        plt.imshow(seedMark, cmap='gray')
        plt.title('region growing')
        plt.axis('off')
        plt.show()

    def regionDivisionMerger(self, img, w0, h0, w, h):
        # 区域分裂和合并
        if self.judge(w0, h0, w, h, img) and (min(w, h) > 5):
            self.regionDivisionMerger(img, w0, h0, int(w / 2), int(h / 2))
            self.regionDivisionMerger(img, w0 + int(w / 2), h0, int(w / 2), int(h / 2))
            self.regionDivisionMerger(img, w0, h0 + int(h / 2), int(w / 2), int(h / 2))
            self.regionDivisionMerger(img, w0 + int(w / 2), h0 + int(h / 2), int(w / 2), int(h / 2))
        else:
            for i in range(w0, w0 + w):
                for j in range(h0, h0 + h):
                    if img[j, i] > 128:
                        img[j, i] = 255
                    else:
                        img[j, i] = 0

    def judge(self, w0, h0, w, h, img):
        # 判断方框是否需要再次拆分为四个
        a = img[h0: h0 + h, w0: w0 + w]
        ave = np.mean(a)
        std = np.std(a, ddof=1)
        count = 0
        total = 0
        for i in range(w0, w0 + w):
            for j in range(h0, h0 + h):
                # 注意！我输入的图片数灰度图，所以直接用的img[j,i]，RGB图像的话每个img像素是一个三维向量，不能直接与avg进行比较大小。
                if abs(img[j, i] - ave) < 1 * std:
                    count += 1
                total += 1
        if (count / total) < 0.95:  # 合适的点还是比较少，接着拆
            return True
        else:
            return False


if __name__ == '__main__':
    s = Segmentation()
    s.segmentation()
    s.regionGrow()
    img = s.image
    s.regionDivisionMerger(img, 0, 0, s.image.shape[0], s.image.shape[1])
    plt.imshow(img, cmap='gray')
    plt.title('Regional division and merger')
    plt.axis('off')
    plt.show()
