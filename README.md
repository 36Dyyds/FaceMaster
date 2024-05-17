# FaceMaster (人脸识别大师)

FaceMaster 是一款基于 Python 的入门级人脸识别项目，旨在为用户提供简单易用的图片和视频人脸识别功能。无论是初学者还是有一定编程经验的用户，都能够轻松上手，通过该项目快速了解和应用人脸识别技术。

## 功能特性

1. 人脸识别：支持图片、视频、摄像头中的人脸识别，使用 opencv、dlib、face_recognition 3种方式实现。
2. 表情识别：包括哀、惊、惧、乐、怒、厌、中七种基本表情。
3. 姓名识别：支持图片、视频、摄像头中的人脸姓名识别。
4. 文字识别：使用 easyocr 进行文字识别，支持中文、英文、数字。

持续关注本项目后续支持更多实现！！！

## 功能演示

### 表情识别

以下演示图片均来源于 https://www.pexels.com 免费下载，如有侵权请告知删除！！！

![表情识别图片](https://gitee.com/qq153128151/FaceMaster/raw/master/output/demo1.jpg)

### 姓名识别

![姓名识别图片](https://gitee.com/qq153128151/FaceMaster/raw/master/output/demo2.jpg)

## 项目环境

- 平台： Windows 10
- 工具：PyCharm 2022.1.2
- Python 版本： 3.6.8
- TensorFlow 版本： 2.6.2
- CUDA 版本：11.2
- cuDNN 版本：8.1.1

TensorFlow 默认不使用 GPU 处理，如需开启请参考 cuda_test.py 请确保 tensorflow 与 Python、CUDA、cuDNN 版本一致。

## 安装依赖

##### 1、克隆本项目

```bash
git clone https://gitee.com/qq153128151/FaceMaster.git
```

或

```bash
git clone https://github.com/36Dyyds/FaceMaster.git
```

##### 2、更换国内镜像

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
pip config get global.index-url
```

##### 3、安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
+-- FaceMaster
    +-- easyocr_models ---------------------------------------------- easyocr模型
        |-- craft_mlt_25k.pth --------------------------------------- 文本检测模型
        |-- zh_sim_g2.pth ------------------------------------------- 文本识别模型
    |-- face_names -------------------------------------------------- 姓名识别素材
    |-- FER2013.zip ------------------------------------------------- 训练数据集
    |-- add_chinese_text_to_image.py -------------------------------- 图片添加中文文本
    |-- cuda_test.py ------------------------------------------------ CUDA 检测
    |-- emotion_detection_cv2.py ------------------------------------ 表情识别 opencv 算法
    |-- emotion_detection_dlib.py ----------------------------------- 表情识别 dlib 算法
    |-- face_detection_cv2.py --------------------------------------- 人脸识别 opencv 算法
    |-- face_detection_dlib.py -------------------------------------- 人脸识别 dlib 算法
    |-- video_face_detection_cv2.py --------------------------------- 视频人脸识别 opencv 算法
    |-- video_face_detection_dlib.py -------------------------------- 视频人脸识别 dlib 算法
    |-- camera_face_detection_cv2.py -------------------------------- 摄像头实时人脸识别 opencv 算法
    |-- camera_face_detection_dlib.py ------------------------------- 摄像头实时人脸识别 dlib 算法
    |-- camera_face_recognition_name.py ----------------------------- 摄像头实时人脸姓名识别 face_recognition 开源库
    |-- face_recognition_demo.py ------------------------------------ 人脸识别 face_recognition 开源库
    |-- face_recognition_name.py ------------------------------------ 人脸姓名识别 face_recognition 开源库
    |-- video_face_recognition_name.py ------------------------------ 视频人脸姓名识别 face_recognition 开源库
    |-- easyocr_demo.py --------------------------------------------- 文字识别 easyocr 开源库
    +-- images ------------------------------------------------------ 测试素材
        |-- 001.jpg
        |-- 002.jpg
        |-- 003.jpg
    +-- models ------------------------------------------------------ 模型
        |-- emotion_detection_model.h5 ------------------------------ 表情识别模型 损失：0.99，准确率：0.63
        |-- haarcascade_frontalface_default.xml --------------------- Haar 级联分类器模型
        |-- shape_predictor_68_face_landmarks.dat ------------------- Dlib 库训练的模型
    +-- output ------------------------------------------------------ 输出文件
        |-- accuracy_plot.png
        |-- loss_plot.png
    +-- videos ------------------------------------------------------ 姓名识别素材
        |-- 001.mp4
    |-- requirements.txt -------------------------------------------- 依赖模块
    |-- train_emotion_model.py -------------------------------------- 训练表情识别模型
```

## 运行说明

项目中的代码都有详细的注释，不依赖其他文件右键运行即可。

表情识别采用 FER2013 数据集，人脸表情模型：测试集损失：0.9903802275657654，测试集准确率：0.6338813304901123

你可以运行 train_emotion_model.py 来训练你的人脸表情模型。

![准确率验证](https://gitee.com/qq153128151/FaceMaster/raw/master/output/accuracy_plot.png)

![损失验证](https://gitee.com/qq153128151/FaceMaster/raw/master/output/loss_plot.png)

## 常见问题

- 部分依赖无法安装：

  dlib==19.7.0 和 face-recognition==1.3.0 不支持python3.6.8以上版本。

- 依赖无法下载：

  使用国内的镜像源，如果开启了代理或科学上网需要关闭。

## 贡献

如果您发现了任何问题或者有任何建议，欢迎提出 issue 或者提交 pull request。

## 群聊

##### 微信交流群

![微信](https://gitee.com/qq153128151/FaceMaster/raw/master/images/wx.png)

扫码添加微信，备注：FaceMaster，邀您加入群聊

## 打赏

如果我的项目对你有所帮助还请给个免费的 Star 能让更多人看到。

![微信打赏](https://gitee.com/qq153128151/FaceMaster/raw/master/images/reward.png)

## 版权信息

本项目遵循 MIT License。详细信息请参阅 LICENSE 文件。

希望 FaceMaster 能够帮助您快速入门和应用人脸识别技术，祝您使用愉快！

