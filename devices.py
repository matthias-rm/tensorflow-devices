import tensorflow as tf
tf.debugging.set_log_device_placement(False)

devices = tf.config.list_logical_devices()
print('All devices:')
for device in devices:
    print(device.name)

for device in devices:
    with tf.device(device.name):
        x = tf.constant([1, 2, 3])
        y = tf.constant([4, 5, 6])
        z = x * y
        tf.assert_equal(z, [4, 10, 18])