import cv2
import dlib


def main():
    # 初始化 Dlib 的人脸检测器和特征点定位器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 读取视频流中的一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 在图像中检测人脸
        faces = detector(frame)

        # 遍历检测到的人脸
        for face in faces:
            # 绘制人脸框
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 获取68个特征点
            shape = predictor(frame, face)
            for pt in shape.parts():
                pt_pos = (pt.x, pt.y)
                # 绘制68个特征点
                cv2.circle(frame, pt_pos, 2, (0, 0, 255), -1)

        # 显示结果
        cv2.imshow('Camera Face Detection', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
