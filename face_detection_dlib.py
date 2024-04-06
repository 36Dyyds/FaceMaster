import cv2
import dlib

# 读取图片
image_path = 'images/003.jpg'
image = cv2.imread(image_path)

# # 缩放图片
# image = cv2.resize(image, None, fx=0.8, fy=0.8)

# 初始化 Dlib 的人脸检测器和特征点定位器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

# 在图像中检测人脸
faces = detector(image, 2)

# 遍历检测到的人脸
for face in faces:
    # 绘制人脸框
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 获取68个特征点
    shape = predictor(image, face)
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        # 绘制68个特征点
        cv2.circle(image, pt_pos, 2, (0, 0, 255), -1)

# 显示结果
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
