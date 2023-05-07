import cv2
import pytesseract
import numpy as np

# 车牌颜色识别阈值范围
lower_blue = np.array([100, 43, 46])
upper_blue = np.array([124, 255, 255])

# 读取图像文件
img = cv2.imread('car.jpg')

# 转换为HSV颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 进行车牌颜色识别
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 进行形态学操作，使车牌区域更清晰
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 查找轮廓
contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的轮廓，即车牌区域
max_contour = max(contours, key=cv2.contourArea)

# 获取车牌区域的外接矩形
rect = cv2.minAreaRect(max_contour)
box = np.int0(cv2.boxPoints(rect))

# 进行透视变换，得到车牌图像
width = int(rect[1][0])
height = int(rect[1][1])
src_pts = box.astype("float32")
dst_pts = np.array([[0, height - 1],
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1]], dtype="float32")
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
plate = cv2.warpPerspective(img, M, (width, height))

# 将车牌区域转换为灰度图像
gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

# 进行二值化处理，使车牌区域中的文字更清晰
ret, thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 使用Tesseract OCR库将车牌区域中的文字转换为文本
text = pytesseract.image_to_string(thresh, lang='chi_sim', config='--psm 7')
print('License Plate: ' + text)

# 显示车牌图像
cv2.imshow('plate', plate)
cv2.waitKey(0)
cv2.destroyAllWindows()
