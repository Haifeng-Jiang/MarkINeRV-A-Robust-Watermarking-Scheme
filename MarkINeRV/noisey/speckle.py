#读取图片
import cv2
import numpy as np
img_height=270
img_width=485
img_channels=3
#读取图片
img = cv2.imread("D:\HNeRV-main\image\\6.png")
#img = cv2.resize(img, (img_height, img_width))
image=img
#随机生成一个服从分布的噪声
gauss = np.random.randn(img_height,img_width,img_channels)
#给图片添加speckle噪声
noisy_img = image + image * gauss
#归一化图像的像素值
noisy_img = np.clip(noisy_img,a_min=0,a_max=255)
#保存图片
cv2.imwrite("D:\HNeRV-main\image\\speckle6.png",noisy_img)
