import tensorflow as tf
import numpy as np
import albumentations as A
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
        self.ids = np.arange(len(self.image_urls))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        indexes = self.ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch.""" #cuz you want each epoch to be randomly shuffle
        if self.shuffle:
            np.random.shuffle(self.ids)

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

    def data_generation(self, indexes):
        """Generate data for the batch."""
        X = np.zeros((len(indexes), *self.img_size, self.input_channels), dtype=np.float32)
        y = np.zeros((len(indexes), *self.img_size, self.n_classes), dtype=np.float32)

        for i, idx in enumerate(indexes):
            img = load_image(self.image_urls[idx])  # Load and preprocess image
            img = np.array(img)
            img = np.transpose(img, (1, 0, 2))  # Convert to WxHxC

            mask = get_mask(img=self.image_urls[idx], labels=self.labels[idx])
            mask = cv2.resize(mask, self.img_size)

            if self.augment:
                aug = self.augmentation()(image=img, mask=mask)
                img = aug['image']
                mask = aug['mask']

            X[i] = img
            y[i] = mask

        X = tf.convert_to_tensor(X, dtype=tf.bfloat16)
        y = tf.convert_to_tensor(y, dtype=tf.bfloat16)

        return X, y
