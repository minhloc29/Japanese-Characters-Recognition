import cv2
import numpy as np
from torch.utils.data import Dataset
from utils import get_mask, load_image
import torch
class JapaneseDataset(Dataset):
    def __init__(self, image_urls, labels, batch_size, input_channels = 3, img_size = (512, 512), n_classes = 2, augment = None, shuffle = True):
        #image_urls: list of image urls as input
        #augment: A_Transform
        self.img_size = img_size
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.image_urls = image_urls
        self.n_classes = n_classes
        self.labels = labels
        self.augment = augment
        self.ids = range(len(self.image_urls))
    def __len__(self):
        return int(len(self.ids) / self.batch_size)
        
    def __getitem__(self, index):
        indexes = list(range(index * self.batch_size, (index+1)*self.batch_size))
        temp_ids = [self.ids[i] for i in indexes]
        X, y = self.data_generation(temp_ids)
        return X, y
    
    def data_generation(self, ids):
        X = np.zeros((0, *self.img_size, self.input_channels))
        y = np.zeros((0, *self.img_size, self.n_classes))
        for i, id in enumerate(self.ids):
            img = load_image(self.image_urls[id]) #tensor, shape: CxHxW
            img = np.array(img)
            img = np.transpose(img, (2, 1, 0)) #convert to WxHxC
            #using [:, :, ::-1] to convert to RGB images cuz cv2 read BGR by default
            mask = get_mask(img = self.image_urls[id], labels= self.labels[id])
            mask = cv2.resize(mask, self.img_size)
            
            if self.augment is not None:
                aug = self.augment(image = img, mask = mask)
                img = aug['image']
                mask = aug['mask']
                
            X = np.vstack((X, np.expand_dims(img, axis=0)))
            y = np.vstack((y, np.expand_dims(mask, axis=0)))
            X = torch.tensor(np.transpose(X, (0, 3, 1, 2)), dtype=torch.float32)
            y = torch.tensor(np.transpose(y, (0, 3, 1, 2)), dtype=torch.float32)
            return X, y
            
        
        