# SRCNN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation

def build_sr_model(scale_factor=2, num_filters=64, num_res_blocks=16):
    # Sequential implementation
    model = Sequential()

    # Convolutional layer
    model.add(Conv2D(num_filters, kernel_size=9, padding='same', input_shape=(None, None, 3)))
    model.add(Activation('relu'))

    # Residual blocks
    for _ in range(num_res_blocks):
        model.add(Conv2D(num_filters, kernel_size=3, padding='same'))
        model.add(Activation('relu'))

    # Convolutional layer
    model.add(Conv2D(num_filters, kernel_size=3, padding='same'))
    model.add(Activation('relu'))

    # Upsampling layer
    for _ in range(scale_factor // 2):
        model.add(Conv2D(num_filters * (scale_factor ** 2), kernel_size=3, padding='same'))
        model.add(Activation('relu'))
    model.add(Conv2D(3, kernel_size=9, padding='same'))

    return model

model = build_sr_model()
model.summary()
