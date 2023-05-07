import cv2


# 读取 BMP 图像文件
image_bmp = cv2.imread('../picture/Fruits.bmp')
print('BMP Image - 5x5 RGB Matrix:')
for i in range(5):
    for j in range(5):
        b, g, r = image_bmp[i][j]  # 获取当前像素点RGB值
        print(f'({r}, {g}, {b})', end=' ')
    print()

# 读取 JPG 图像文件
image_jpg = cv2.imread('../picture/Fruits.jpg')
print('JPG Image - 5x5 RGB Matrix:')
for i in range(5):
    for j in range(5):
        b, g, r = image_jpg[i][j]  # 获取当前像素点RGB值
        print(f'({r}, {g}, {b})', end=' ')
    print()
