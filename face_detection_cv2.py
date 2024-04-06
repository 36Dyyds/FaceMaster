import cv2

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# 读取图片
image_path = 'images/003.jpg'
image = cv2.imread(image_path)

# # 缩放图片
# image = cv2.resize(image, None, fx=0.8, fy=0.8)

# 检测人脸
faces = face_cascade.detectMultiScale(image)

# 在图像中标记人脸
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 显示图像
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
