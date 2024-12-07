from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from torchvision import transforms
import torch
def get_mask(img, labels):
    # input: 
    #     img: url
    #     label: str
    img = Image.open(img).convert('RGBA')
    img = np.array(img)
    mask = np.zeros((img.shape[0], img.shape[1], 2), dtype='float32')
    if isinstance(labels, str):
        labels = np.array(labels.split(' ')).reshape(-1, 5)
        for char, x, y, w, h in labels:
            x, y, w, h = int(x), int(y), int(w), int(h)
            if x + w >= img.shape[1] or y + h >= img.shape[0]:
                continue
            mask[y: y + h, x: x + w, 0] = 1 #giving 1 to region that are 1
            radius = 6
            mask[y + h // 2 - radius: y + h // 2 + radius + 1, x +
                 w // 2 - radius: x + w // 2 + radius + 1, 1] = 1 #giving 1 to a whole region near the center-point
    return mask

def load_image(img_url, img_size = (512, 512), expand_dim = False):
    #input: image_url, output: tensor, shape: cxhxw
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to Tensor, scaling values to [0, 1]
    transforms.Normalize(mean=mean, std=std),  # Normalizes each channel (R, G, B)
])
    img = cv2.imread(img_url)[:, :, ::-1] #conver to RGB, cuz normally cv2.imread reads the BGR
    h, w, c = img.shape
    img = cv2.resize(img, img_size)
    img = Image.fromarray(img)
    img = transform(img)
    if expand_dim:
        img = torch.unsqueeze(0)
    return img
