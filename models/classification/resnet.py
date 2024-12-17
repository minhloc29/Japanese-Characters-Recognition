import tensorflow as tf
from tensorflow.keras import layers, Model

def batch_activate(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def convolution_block(x,
                      filters,
                      size,
                      strides=(1, 1),
                      padding='same',
                      activation=True):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = batch_activate(x)
    return x


def residual_block(block_input,
                   num_filters=16,
                   use_batch_activate=False):
    x = batch_activate(block_input)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = layers.Add()([x, block_input])
    if use_batch_activate:
        x = batch_activate(x)
    return x

def resnet_backbone(no_classes=2028,
                    no_channels=3,
                    start_neurons=32,
                    dropout_rate=0.1):
    input_layer = layers.Input(
        name='input_image',
        shape=(64, 64, no_channels),
        dtype='float32'
    )

    for index, i in enumerate([1, 2, 2, 4, 8]):
        if index == 0:
            inner = input_layer
        inner = layers.Conv2D(start_neurons * i, (3, 3),
                              activation=None, padding="same")(inner)
        inner = residual_block(inner, start_neurons * i)
        inner = residual_block(inner, start_neurons * i, True)

        if i <= 4:
            inner = layers.MaxPooling2D((2, 2))(inner)

        if dropout_rate:
            inner = layers.Dropout(dropout_rate)(inner)

    inner = layers.Flatten()(inner)
    inner = layers.Dense(no_classes, activation="softmax")(inner)
    net = Model(inputs=[input_layer], outputs=inner)
    return net
