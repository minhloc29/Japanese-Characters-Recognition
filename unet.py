import tensorflow as tf
from tensorflow.keras import layers, Model

# Residual Block
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, use_relu=True, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.use_relu = use_relu
        self.conv1 = layers.Conv2D(filters, kernel_size=3, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size=3, padding="same")
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs, training=False):
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        if self.use_relu:
            x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        if self.use_relu:
            x = tf.nn.relu(x)
        return x + residual

# ResNet-UNet
class ResNetUNet(Model):
    def __init__(self, img_size=(512, 512), no_channels=3, start_neurons=32, dropout_rate=0.25, **kwargs):
        super(ResNetUNet, self).__init__(**kwargs)
        self.dropout = layers.Dropout(dropout_rate)
        # Downsampling
        self.conv1 = layers.Conv2D(start_neurons * 1, kernel_size=3, padding="same")
        self.block1 = [
            ResidualBlock(start_neurons * 1, use_relu=True),
            ResidualBlock(start_neurons * 1, use_relu=False)
        ]
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2 = layers.Conv2D(start_neurons * 2, kernel_size=3, padding="same")
        self.block2 = [
            ResidualBlock(start_neurons * 2, use_relu=True),
            ResidualBlock(start_neurons * 2, use_relu=False)
        ]
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv3 = layers.Conv2D(start_neurons * 4, kernel_size=3, padding="same")
        self.block3 = [
            ResidualBlock(start_neurons * 4, use_relu=True),
            ResidualBlock(start_neurons * 4, use_relu=False)
        ]
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2))

        # Middle
        self.middle_conv = layers.Conv2D(start_neurons * 8, kernel_size=3, padding="same")
        self.middle_block = [
            ResidualBlock(start_neurons * 8, use_relu=True),
            ResidualBlock(start_neurons * 8, use_relu=False)
        ]

        # Upsampling
        self.deconv3 = layers.Conv2DTranspose(start_neurons * 4, kernel_size=3, strides=2, padding="same")
        self.uconv3 = layers.Conv2D(start_neurons * 4, kernel_size=3, padding="same")
        self.block_u3 = [
            ResidualBlock(start_neurons * 4, use_relu=True),
            ResidualBlock(start_neurons * 4, use_relu=False)
        ]

        self.deconv2 = layers.Conv2DTranspose(start_neurons * 2, kernel_size=3, strides=2, padding="same")
        self.uconv2 = layers.Conv2D(start_neurons * 2, kernel_size=3, padding="same")
        self.block_u2 = [
            ResidualBlock(start_neurons * 2, use_relu=True),
            ResidualBlock(start_neurons * 2, use_relu=False)
        ]

        self.deconv1 = layers.Conv2DTranspose(start_neurons * 1, kernel_size=3, strides=2, padding="same")
        self.uconv1 = layers.Conv2D(start_neurons * 1, kernel_size=3, padding="same")
        self.block_u1 = [
            ResidualBlock(start_neurons * 1, use_relu=True),
            ResidualBlock(start_neurons * 1, use_relu=False)
        ]
        # Output
        self.output_layer = layers.Conv2D(2, kernel_size=1, padding="same", activation="sigmoid")

    def call(self, inputs, training=False):
        # Downsampling
        conv1 = self.conv1(inputs)
        for block in self.block1:
            conv1 = block(conv1, training=training)
        pool1 = self.dropout(self.pool1(conv1), training=training)

        conv2 = self.conv2(pool1)
        for block in self.block2:
            conv2 = block(conv2, training=training)
        pool2 = self.dropout(self.pool2(conv2), training=training)

        conv3 = self.conv3(pool2)
        for block in self.block3:
            conv3 = block(conv3, training=training)
        pool3 = self.dropout(self.pool3(conv3), training=training)

        # Middle
        middle = self.middle_conv(pool3)
        for block in self.middle_block:
            middle = block(middle, training=training)

        # Upsampling
        deconv3 = self.deconv3(middle)
        uconv3 = tf.concat([deconv3, conv3], axis=-1)
        uconv3 = self.uconv3(uconv3)
        for block in self.block_u3:
            uconv3 = block(uconv3, training=training)

        deconv2 = self.deconv2(uconv3)
        uconv2 = tf.concat([deconv2, conv2], axis=-1)
        uconv2 = self.uconv2(uconv2)
        for block in self.block_u2:
            uconv2 = block(uconv2, training=training)

        deconv1 = self.deconv1(uconv2)
        uconv1 = tf.concat([deconv1, conv1], axis=-1)
        uconv1 = self.uconv1(uconv1)
        for block in self.block_u1:
            uconv1 = block(uconv1, training=training)

        # Output
        output = self.output_layer(uconv1)
        return output
