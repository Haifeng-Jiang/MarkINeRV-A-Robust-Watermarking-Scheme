import cv2
img = cv2.imread("D:\HNeRV-main\image\\6.png")
dst_size = (485, 270)
img_resize = cv2.resize(img, dst_size, interpolation = cv2.INTER_AREA)
cv2.imwrite("D:\HNeRV-main\image\\6.png",img_resize)
print(img_resize.shape)
