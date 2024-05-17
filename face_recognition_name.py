import os
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def main():
    # 读取图片
    image_path = 'images/002.jpg'
    image = cv2.imread(image_path)

    # 缩放图片
    # image = cv2.resize(image, None, fx=0.8, fy=0.8)

    # 加载人脸图片及其对应的中文名字
    image_dir = 'face_names'
    face_images = os.listdir(image_dir)
    face_encodings = []
    face_names = []
    for face_image in face_images:
        # 获取图片的名称作为姓名
        name, _ = os.path.splitext(face_image)
        face_names.append(name)

        image_file = face_recognition.load_image_file(os.path.join(image_dir, face_image))
        face_encoding = face_recognition.face_encodings(image_file)[0]
        face_encodings.append(face_encoding)

    # 在图片中查找人脸
    face_locations = face_recognition.face_locations(image)
    face_encodings_in_image = face_recognition.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_in_image):
        # 比较人脸编码
        matches = face_recognition.compare_faces(face_encodings, face_encoding)

        name = '未知'
        if True in matches:
            first_match_index = matches.index(True)
            name = face_names[first_match_index]

        # 绘制人脸框
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        image = add_chinese_text(image, name, position=(left, top - 35))

    # 显示结果
    cv2.imshow('Face Name Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 将图像保存到本地
    output_path = 'output/004.jpg'
    # 确保输出文件夹存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 保存图像
    cv2.imwrite(output_path, image)


if __name__ == '__main__':
    main()
