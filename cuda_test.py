import sys
import timeit
import tensorflow as tf

"""
tensorflow-gpu与Python、CUDA、cuDNN 版本关系查询
https://tensorflow.google.cn/install/source_windows?hl=zh-cn#gpu
CUDA 下载地址
https://developer.nvidia.com/cuda-toolkit-archive
cuDNN 下载地址
https://developer.nvidia.com/rdp/cudnn-archive

平台： Windows 10
Python 版本： 3.6.8
TensorFlow 版本： 2.6.2
CUDA 版本：11.2
cuDNN 版本：8.1.1
"""

print('Python 版本：', sys.version)
print("TensorFlow版本：", tf.__version__)
print("可用GPU物理设备：", tf.config.list_physical_devices('GPU'))
print("是否使用CUDA构建：", tf.test.is_built_with_cuda)
print("当前可用GPU设备名：", tf.test.gpu_device_name())
print("当前可见设备列表：", tf.config.get_visible_devices())
print("系统中是否有可用GPU：", tf.test.is_gpu_available())


# 指定在cpu上运行
def cpu_run():
    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([6000, 6000])
        cpu_b = tf.random.normal([6000, 6000])
        c = tf.matmul(cpu_a, cpu_b)
    return c


# 指定在gpu上运行
def gpu_run():
    with tf.device('/gpu:0'):
        gpu_a = tf.random.normal([6000, 6000])
        gpu_b = tf.random.normal([6000, 6000])
        c = tf.matmul(gpu_a, gpu_b)
    return c


if __name__ == '__main__':
    if tf.test.is_gpu_available():
        cpu_time = timeit.timeit(cpu_run, number=10)
        gpu_time = timeit.timeit(gpu_run, number=10)
        print('CPU：', cpu_time, 'GPU：', gpu_time)
    else:
        print('未安装CUDA或没有GPU')
