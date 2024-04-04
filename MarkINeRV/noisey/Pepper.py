import cv2
import numpy as np
#读取图片
img_height=155
img_width=165

#读取图片
img = cv2.imread("D:\HNeRV-main\image\\11.png")
#img = cv2.resize(img, (img_height, img_width))
image=img
#设置添加椒盐噪声的数目比例
s_vs_p = 0.5
#设置添加噪声图像像素的数目
amount = 0.04
noisy_img = np.copy(image)
#添加salt噪声
num_salt = np.ceil(amount * image.size * s_vs_p)
#设置添加噪声的坐标位置
coords = [np.random.randint(0,i - 1, int(num_salt)) for i in image.shape]
noisy_img[coords[0],coords[1],:] = [255,255,255]
#添加pepper噪声
num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
#设置添加噪声的坐标位置
coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in image.shape]
noisy_img[coords[0],coords[1],:] = [0,0,0]
#保存图片
cv2.imwrite("D:\HNeRV-main\image\\jy11.png",noisy_img)
