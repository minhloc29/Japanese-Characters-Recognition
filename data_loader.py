import tensorflow as tf
import numpy as np
import cv2
import albumentations as A
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

    def augmentation(self):
        """Define the augmentation pipeline using Albumentations."""
        return A.Compose([
            A.RandomCrop(width=512, height=512, p=0.75),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(
                p=0.5, rotate_limit=1.5,
                scale_limit=0.05, border_mode=0
            )
        ])

    def fixup_shape(self, images, mask):
        images.set_shape([*self.img_size, self.input_channels])
        mask.set_shape([*self.img_size, self.n_classes])
        return images, mask
    
    def process_data(self, image_url, label):
        # image_url, label: String Tensor
        image_url = image_url.numpy().decode('utf-8')
        label = label.numpy().decode('utf-8')
        image = cv2.imread(image_url) #original image
        mask = get_mask(img=image, labels=label)  # numpy: 512x512x2
        image = load_image(image_url)  # numpy: W x H x C, 512x512x3
        
        if self.augment:
            aug = self.augmentation()(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        return image, mask
    
    def create_tf_dataset(self):
        
        dataset = tf.data.Dataset.from_tensor_slices((self.image_urls, self.labels))

        # Map the process_data function to the dataset
        dataset = dataset.map(lambda img_url, label: tf.py_function(self.process_data, [img_url, label], [tf.float32, tf.float32]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.fixup_shape)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.repeat() #repeat the dataset infinitely
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset

