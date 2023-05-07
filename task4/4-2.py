import cv2
import numpy as np
import base64
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.ocr.v20181119 import ocr_client, models

# 读取图像文件
img = cv2.imread('car.jpg')

# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用高斯滤波去除噪声
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 应用Canny边缘检测
edges = cv2.Canny(gray, 100, 200)

# 使用形态学操作，使车牌区域更清晰
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# 查找轮廓
contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的轮廓，即车牌区域
max_contour = max(contours, key=cv2.contourArea)

# 创建掩膜图像，用于提取车牌区域
mask = np.zeros(gray.shape, dtype=np.uint8)
cv2.drawContours(mask, [max_contour], -1, 255, -1)

# 提取车牌区域
result = cv2.bitwise_and(img, img, mask=mask)

# 将车牌区域转换为灰度图像
gray_plate = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# 应用阈值处理，使车牌区域中的文字更清晰
ret, thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 使用腾讯OCR SDK将车牌区域中的文字转换为文本
try:
    cred = credential.Credential("YourSecretId", "YourSecretKey")
    client = ocr_client.OcrClient("ap-guangzhou", cred)

    req = models.GeneralBasicOCRRequest()
    req.ImageBase64 = str(base64.b64encode(cv2.imencode('.jpg', thresh)[1]).decode('utf-8'))
    req.LanguageType = "zh"
    response = client.GeneralBasicOCR(req)
    print("License Plate: " + response.TextDetections[0].DetectedText)
except TencentCloudSDKException as err:
    print(err)

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
