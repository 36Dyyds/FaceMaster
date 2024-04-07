import cv2


def main():
    # 加载视频
    video_path = 'videos/001.mp4'
    video_capture = cv2.VideoCapture(video_path)

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    while True:
        # 读取视频帧
        ret, image = video_capture.read()
        if not ret:
            break

        # 缩放图片
        # image = cv2.resize(image, None, fx=0.8, fy=0.8)

        # 检测人脸
        faces = face_cascade.detectMultiScale(image)

        # 在图像中标记人脸
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 显示图像
        cv2.imshow('Video Face Detection', image)
        # 退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
