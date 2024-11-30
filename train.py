import tensorflow as tf
import time
from utils import get_model, get_datasets

x_train, y_train, x_test, y_test = get_datasets()

devices =  tf.config.list_logical_devices()
print('All devices:')
for device in devices:
    print(device.name)

for device in devices:
    print('Training on: ', device.name)
    with tf.device(device.name):
        model = get_model(x_train[0].shape, num_classes=10)
        # Compile
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        start = time.time()
        r = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test), epochs=1)
        end = time.time()
    print(f'Device {device.name} took {end - start:.2f}s')