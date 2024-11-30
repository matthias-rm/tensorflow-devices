import tensorflow as tf
from tensorflow.python.client import device_lib
import time
from utils import get_model, get_datasets
for device in device_lib.list_local_devices(): print(device.name)

x_train, y_train, x_test, y_test = get_datasets()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

for device in device_lib.list_local_devices():
    model = get_model(x_train[0].shape, num_classes=10)
    # Compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    with tf.device(device.name):
        start = time.time()
        r = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test), epochs=2)
        end = time.time()
    print(f'Device {device.name}: {end - start:.2f}s')

# CIFAR 10
# ENV wi/o tensorflow-metal, /device:CPU:0: 126.11s - 147.74s
# ENV with tensorflow-metal, /device:CPU:0: 155.79s
# ENV with tensorflow-metal, /device:GPU:0:  76.39s
# CTR
# CPU /device:CPU:0: 223s
# GPU /device:GPU:0: 331s
