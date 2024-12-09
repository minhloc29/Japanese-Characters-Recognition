import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv, json
import cv2

def visualize_training_data(image_fn,
                            labels,
                            width=3,
                            y_first=False, FONT_SIZE = 100):
    
    unicode_map_json = "data/unicode_map.json"
    with open(unicode_map_json, 'r') as f:
        unicode_map = json.load(f)
        
    labels = np.array(labels.split(' ')).reshape(-1, 5) #divide to list of each bounding box
    imsource = Image.open(image_fn).convert('RGBA')
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)

    bbox_draw = ImageDraw.Draw(bbox_canvas)
    char_draw = ImageDraw.Draw(char_canvas)

    for codepoint, *args in labels: 
        if y_first:
            y, x, h, w = args
        else:
            x, y, w, h = args

        x, y, w, h = int(x), int(y), int(w), int(h)
        try:
            char = unicode_map[codepoint]
        except KeyError:
            # some codepoint not exists in unicode_map :/
            print(codepoint)
            continue
        # Draw bounding box around character, and unicode character next to it
        bbox_draw.rectangle(
            (x, y, x + w, y + h), fill=(255, 255, 255, 0),
            outline=(255, 0, 0, 255), width=width
        )
        char_draw.text(
            (x + w + FONT_SIZE / 4, y + h / 2 - FONT_SIZE),
            char, fill=(0, 0, 255, 255),
            font = ImageFont.truetype("NotoSansJP-Bold.ttf", FONT_SIZE)
        )
    imsource = Image.alpha_composite(
        Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    # Remove alpha for saving in jpg format.
    imsource = imsource.convert("RGB")
    return np.asarray(imsource)
                                
def get_mask(img, labels, img_size = (512, 512)):
    #img: input_shape: W x H x C, original shape without resizing
    mask = np.zeros((img.shape[0], img.shape[1], 2), dtype='float32')
    if isinstance(labels, str):
        labels = np.array(labels.split(' ')).reshape(-1, 5)
        for char, x, y, w, h in labels:
            x, y, w, h = int(x), int(y), int(w), int(h)
            if x + w >= img.shape[1] or y + h >= img.shape[0]:
                continue
            # Mark region with 1
            mask[y:y + h, x:x + w, 0] = 1  
            # Mark center region with 1
            radius = 6
            mask[y + h // 2 - radius: y + h // 2 + radius + 1, 
                 x + w // 2 - radius: x + w // 2 + radius + 1, 1] = 1
    mask = cv2.resize(mask, img_size)
    return mask #numpy: W x H x C

def load_image(img_url, img_size=(512, 512), expand_dim=False):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # Reshape to (1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    # Load the image using PIL
    img = Image.open(img_url).convert("RGB")
    img = img.resize(img_size)  
    img = np.array(img) / 255.0 
    # Normalize the image using mean and std
    img = (img - mean) / std 
    if expand_dim:
        img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img #numpy: 512x512x3
