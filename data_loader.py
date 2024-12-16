import tensorflow as tf
import numpy as np
import cv2
import albumentations as A
import random
from utils import get_mask, load_image, resize_padding
#dataset for segmentation
class JapaneseDataset:
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
        mask = get_mask(img=image, labels=label)  
        mask = cv2.resize(mask, self.img_size) # numpy: 512x512x2
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

#dataset for classification
class ClassifierDataset:
    def __init__(self, image_urls, labels, batch_size, input_channels=3, img_size=(64, 64), augment = False, shuffle = False):
        self.image_urls = image_urls
        self.labels = labels
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.size = img_size[0]
        self.num_classes = 2028
    def augmentation(self):
        return A.Compose([
        A.RandomBrightnessContrast(
            contrast_limit=0.2, brightness_limit=0.2, p=0.5
        ),  # Adjust lighting variations
        A.Blur(blur_limit=3, p=0.2),             # Simulate minor blur
        A.Rotate(limit=5, p=0.5),                # Small rotations for robustness
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=30, p=0.3),  # Distortions for handwriting
    ])
        
    def fixup_shape(self, image, label):
        image.set_shape([*self.img_size, 3])
        label.set_shape([self.num_classes])
        return image, label
    
    def process_data(self, image_url, label):
        image_url = image_url.numpy().decode('utf-8')
        label = label.numpy()
        label = np.array(label, dtype=np.int32).reshape(1)
        one_hot_label = np.eye(self.num_classes)[label].reshape(-1)
        label = np.array(label, dtype=np.int32).reshape(1)
        image = cv2.imread(image_url)[:,:, ::-1]
        image = image / 255
        image = image.astype('float32')

        mean = np.mean(image, axis=(0, 1, 2))
        std = np.std(image, axis=(0, 1, 2))
        image = (image - mean) / std
        image = resize_padding(image, self.size)
        if self.augment:
            aug = self.augmentation()(image=image)
            image = aug['image']
        #normalize data
        return image, one_hot_label
    
    def create_tf_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_urls, self.labels))
        dataset = dataset.map(lambda img_url, label: tf.py_function(self.process_data, [img_url, label], [tf.float32, tf.int32]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.fixup_shape)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
