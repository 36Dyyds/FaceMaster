import cv2


def main():
    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 读取视频流中的一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 检测人脸
        faces = face_cascade.detectMultiScale(frame)

        # 在图像中标记人脸
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 显示图像
        cv2.imshow('Camera Face Detection', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
