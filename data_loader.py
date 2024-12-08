import tensorflow as tf
import numpy as np
import cv2
from utils import get_mask, load_image

class JapaneseDataset(tf.keras.utils.Sequence):
    def __init__(self, image_urls, labels, batch_size, input_channels=3, img_size=(512, 512), n_classes=2, augment=False, shuffle=True):
       
        self.img_size = img_size
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.image_urls = image_urls
        self.n_classes = n_classes
        self.labels = labels
        self.augment = augment
        self.shuffle = shuffle

    def augmentation(self, image, mask):
        
        if tf.random.uniform(()) < 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
        
        angles = [0, 90, 180, 270]
        angle_idx = tf.random.uniform([], minval=0, maxval=len(angles), dtype=tf.int32)
        angle = angles[angle_idx.numpy()]  # Get the angle using the index

        image = tf.image.rot90(image, k=angle // 90)
        mask = tf.image.rot90(mask, k=angle // 90)

        return image, mask

    def process_data(self, image_url, label):
        image = load_image(image_url)  # Tensor: W x H x C, 512x512x3
        mask = get_mask(img=image_url, labels=label)  # numpy: W x H x C
        mask = tf.convert_to_tensor(mask)
        mask = tf.image.resize(mask, self.img_size)

        if self.augment:
            image, mask = self.augmentation(image, mask)

        return image, mask
    
    def create_tf_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_urls, self.labels))

        # Map the process_data function to the dataset
        dataset = dataset.map(lambda img_url, label: tf.py_function(self.process_data, [img_url, label], [tf.float32, tf.float32]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset

