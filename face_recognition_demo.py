import cv2
import face_recognition

# 读取图片
image_path = 'images/002.jpg'
image = cv2.imread(image_path)

# # 缩放图片
# image = cv2.resize(image, None, fx=0.8, fy=0.8)

# 在图像中检测人脸
face_locations = face_recognition.face_locations(image)

# 绘制人脸框
for top, right, bottom, left in face_locations:
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

# 在图像中标识人脸关键点
face_landmarks_list = face_recognition.face_landmarks(image)
for face_landmarks in face_landmarks_list:
    for facial_feature in face_landmarks.keys():
        for point in face_landmarks[facial_feature]:
            cv2.circle(image, point, 2, (255, 0, 0), -1)

# 显示结果
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
