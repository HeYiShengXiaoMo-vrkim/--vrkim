import tensorflow as tf
print("当前 TensorFlow 版本:", tf.__version__)
print("检测到GPU设备:", tf.config.list_physical_devices('GPU'))