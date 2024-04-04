import cv2
import numpy as np
img_height=640
img_width=640
img_channels=3
#读取图片
img = cv2.imread("D:\HNeRV-main\image\\00000.png")
#img = cv2.resize(img, (img_height, img_width))
#设置高斯分布的均值和方差
mean = 0
#设置高斯分布的标准差
sigma = 25
#根据均值和标准差生成符合高斯分布的噪声
gauss = np.random.normal(mean,sigma,(img_height,img_width,img_channels))
#给图片添加高斯噪声
noisy_img = img + gauss
#设置图片添加高斯噪声之后的像素值的范围
noisy_img = np.clip(noisy_img,a_min=0,a_max=255)
#保存图片
cv2.imwrite("D:\HNeRV-main\image\\gs00000.png",noisy_img)
