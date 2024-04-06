import os
import cv2
import dlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model


def add_chinese_text(image, text, font_path='font/simsun.ttc', font_size=30, font_color=(0, 255, 0), position=(10, 10)):
    """
    在图像上添加中文文本。

    参数：
    - image: 要添加文本的图像，OpenCV 格式的图像数组。
    - text: 要添加的文本内容。
    - font_path: 中文字体文件的路径，默认为 'font/simsun.ttc'。
    - font_size: 字体大小，默认为 30。
    - font_color: 字体颜色，默认为绿色。
    - position: 文本左上角的坐标，默认为 (10, 10)。
    """
    # 转换图像格式为 RGB（PIL 要求图像格式为 RGB）
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 创建绘图对象
    draw = ImageDraw.Draw(pil_image)

    # 加载中文字体
    font = ImageFont.truetype(font_path, font_size)

    # 在图像上绘制文本
    draw.text(position, text, font=font, fill=font_color)

    # 将 PIL 图像转换回 OpenCV 格式
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def detect_emotion(image, model, detector, predictor, confidence_threshold=0.6):
    # 定义表情类别
    emotion_labels = ['哀', '惊', '惧', '乐', '怒', '厌', '中']
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 在图像中检测人脸
    faces = detector(image, 2)
    # 遍历检测到的人脸
    for face in faces:
        # 获取检测到的人脸
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        # 对检测到的每张人脸进行情绪识别
        face_roi = gray[y:y + h, x:x + w]
        resized_face = cv2.resize(face_roi, (48, 48))
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
        result = model.predict(reshaped_face)

        # 获取最大置信度
        max_confidence = np.max(result)
        # 获取情绪标签
        label = np.argmax(result)

        # 置信度高于60%
        if max_confidence > confidence_threshold:
            # 绘制人脸框
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 绘制情绪标签和置信度
            emotion = f'{emotion_labels[label]}：{max_confidence * 100:.2f}%'
            image = add_chinese_text(image, emotion, position=(x, y - 35))

            # 获取68个特征点
            shape = predictor(image, face)
            for pt in shape.parts():
                pt_pos = (pt.x, pt.y)
                # 绘制68个特征点
                cv2.circle(image, pt_pos, 2, (0, 0, 255), -1)
        else:
            print(f'表情：{emotion_labels[label]}，置信度：{max_confidence * 100:.2f}%')

    return image


def main():
    # 读取图片
    image_path = 'images/003.jpg'
    image = cv2.imread(image_path)

    # # 缩放图片
    # image = cv2.resize(image, None, fx=0.8, fy=0.8)

    # 初始化 Dlib 的人脸检测器和特征点定位器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

    # 加载训练好的模型
    model = load_model('models/emotion_detection_model.h5')

    # 进行情绪识别
    result_image = detect_emotion(image, model, detector, predictor)

    # 显示结果
    cv2.imshow('Emotion Detection', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 将图像保存到本地
    output_path = 'output/003.jpg'
    # 确保输出文件夹存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 保存图像
    cv2.imwrite(output_path, result_image)


if __name__ == "__main__":
    main()
