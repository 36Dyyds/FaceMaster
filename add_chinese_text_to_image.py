import os
import cv2
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
    image_path = 'images/001.jpg'
    image = cv2.imread(image_path)

    # 检查是否成功读取图片
    if image is not None:
        # # 缩放图片
        # image = cv2.resize(image, None, fx=0.8, fy=0.8)

        # 在图像上添加中文文本
        text = '你好，PIL！'
        cv2_image_with_text = add_chinese_text(image, text, font_size=80)

        # 显示图片
        cv2.imshow('Image with Chinese Text', cv2_image_with_text)
        cv2.waitKey(0)  # 等待键盘输入，0 表示持续等待直到按下任意键
        cv2.destroyAllWindows()  # 关闭所有窗口

        # 将图像保存到本地
        output_path = 'output/001.jpg'
        # 确保输出文件夹存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 保存图像
        cv2.imwrite(output_path, cv2_image_with_text)
    else:
        print('无法加载图像：', image_path)


if __name__ == "__main__":
    main()
