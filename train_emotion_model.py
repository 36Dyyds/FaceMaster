import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from skimage import io
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 定义表情类别
emotion_labels = ['哀', '惊', '惧', '乐', '怒', '厌', '中']


# 读取图像和创建标签
def load_data(data_dir, subset):
    images = []
    labels = []
    for label, emotion in enumerate(emotion_labels):
        sub_dir = os.path.join(data_dir, subset, emotion)
        for image_name in os.listdir(sub_dir):
            image_path = os.path.join(sub_dir, image_name)
            image = io.imread(image_path)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)


# 划分数据集
def split_data(images, labels, train_ratio=0.8, val_ratio=0.1):
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=1 - train_ratio, random_state=42)
    val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=val_ratio / (1 - train_ratio), random_state=42)
    return train_images, train_labels, val_images, val_labels, test_images, test_labels


# 数据预处理
def preprocess_images(images):
    return images.astype('float32') / 255.0


# 构建模型
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# 绘制损失曲线和准确率曲线
def plot_history(history):
    # 指定中文字体
    font = FontProperties(family='SimSun', size=14)  # 使用宋体字体

    # 绘制损失曲线
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel('轮次', fontproperties=font)
    plt.ylabel('损失', fontproperties=font)
    plt.title('训练和验证损失', fontproperties=font)
    plt.legend(prop=font)

    # 将图像保存到本地
    save_path = 'output/loss_plot.png'
    # 确保输出文件夹存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

    # 绘制准确率曲线
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.xlabel('轮次', fontproperties=font)
    plt.ylabel('准确率', fontproperties=font)
    plt.title('训练和验证准确率', fontproperties=font)
    plt.legend(prop=font)

    # 将图像保存到本地
    save_path = 'output/accuracy_plot.png'
    # 确保输出文件夹存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()


# 主方法
def main():
    # 数据准备
    data_dir = 'FER2013'  # FER2013数据集所在的路径
    train_images, train_labels = load_data(data_dir, '训练')
    test_images, test_labels = load_data(data_dir, '测试')
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    # 调整数据维度
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    # 构建模型
    input_shape = (48, 48, 1)  # 图像大小为48x48，灰度图像
    num_classes = 7  # 表情类别数
    model = build_model(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 数据增强
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
    datagen.fit(train_images)

    # 设置早停策略
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=64), steps_per_epoch=len(train_images) // 64, epochs=500, validation_data=(test_images, test_labels), callbacks=[early_stopping])

    # 调用函数绘制曲线并保存到指定目录
    plot_history(history)

    print(f"训练集损失：{history.history['loss']}")
    print(f"验证集损失：{history.history['val_loss']}")
    print(f"训练集准确率：{history.history['accuracy']}")
    print(f"验证集准确率：{history.history['val_accuracy']}")

    # 评估模型
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'测试集损失：{test_loss}，测试集准确率：{test_acc}')

    # 保存模型到 models 目录下
    model_path = 'models/emotion_detection_model.h5'
    model.save(model_path)
    print(f'模型已保存为 {model_path}')


# 如果作为主模块运行，则调用主方法
if __name__ == "__main__":
    main()
