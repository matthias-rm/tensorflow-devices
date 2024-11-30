import tensorflow as tf
from tensorflow.python.client import device_lib
tf.debugging.set_log_device_placement(True)

print('All devices:')
for device in device_lib.list_local_devices():
    print(device.name)


with tf.device('GPU:0'):
    x = tf.constant([1, 2, 3])
    y = tf.constant([4, 5, 6])
    z = x * y
    print(z)

# for device in device_lib.list_local_devices():
#     with tf.device(device.name):
#         x = tf.constant([1, 2, 3])
#         y = tf.constant([4, 5, 6])
#         z = x @ y
#         print(z)