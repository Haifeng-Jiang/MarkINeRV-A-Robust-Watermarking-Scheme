import cv2
import numpy as np
img_height=155
img_width=165
img_channels=3
#读取图片
img = cv2.imread("D:\HNeRV-main\image\\11111.png")
#img = cv2.resize(img, (img_height, img_width))
image=img
#计算图像像素的分布范围
vals = len(np.unique(image))
vals = 2 ** np.ceil(np.log2(vals))
#给图片添加泊松噪声
noisy_img = np.random.poisson(image * vals) / float(vals)
#保存图片
cv2.imwrite("D:\HNeRV-main\image\\bs11111.png",noisy_img)
