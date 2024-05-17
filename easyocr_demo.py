import easyocr


def perform_ocr(image_path):
    # 指定模型存储目录
    custom_model_directory = 'easyocr_models'

    # 创建 EasyOCR 对象，并指定模型存储目录
    reader = easyocr.Reader(['ch_sim', 'en'], model_storage_directory=custom_model_directory)

    # 执行文字识别
    result = reader.readtext(image_path)

    # 解析识别结果
    recognized_text = ''.join([entry[1] for entry in result])

    return recognized_text


if __name__ == '__main__':
    # 图像文件路径
    image_path = 'output/001.jpg'

    # 执行文字识别
    recognized_text = perform_ocr(image_path)

    # 打印识别结果
    print(f'识别结果：{recognized_text}')
