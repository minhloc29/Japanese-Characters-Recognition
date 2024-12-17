#This code recreate the process of cropping images to multiple small labels
count_url = 0
from my_utils import load_image, get_mask, visualize_training_data
import pandas as pd
import numpy as np
import os
from PIL import Image
import cv2
import json
from tqdm.notebook import tqdm
import csv
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("data/train.csv")
relative_image_urls = train['image_url'].to_list()
labels = train['labels'].to_list()
image_urls = [os.path.join("data/train_images", relative_image_url) for relative_image_url in relative_image_urls]

with open("data/unicode_map.json", 'r') as f:
    unicode_map = json.load(f)
    
def crop_image(image_url: str, label_str: str):
    global count_url
    image = cv2.imread(image_url)
    unicodes = []
    chars = []
    file_names = []
    image_height, image_width, _ = image.shape
    labels = np.array(label_str.split(' ')).reshape(-1, 5)
    for i, (unicode, x, y, w, h) in enumerate(labels):
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        pad_x = int(0.05*w)
        pad_y = int(0.05*h)
        
        start_x = max(0, x - pad_x)
        start_y = max(0, y - pad_y)
        end_x = min(x + w + pad_x, image_width)
        end_y = min(y + h + pad_y, image_height)
        
        cropped_image = image[start_y:end_y, start_x:end_x]
        file_name = f"data/character_images/{unicode}_{count_url + 1}.jpg"
        try:   
            cv2.imwrite(file_name, cropped_image)
            char = unicode_map[unicode]
            chars.append(char)
            unicodes.append(unicode)
            file_names.append(file_name)
        except Exception as e:
            print("Can not process this sample!")
            continue
        count_url += 1
    return file_names, unicodes, chars
    
file_name = "data/classifier.csv"
file_exists = os.path.exists(file_name)

for i, (image_url, label_str) in tqdm(enumerate(zip(image_urls, labels))):
    file_names, unicodes, chars = crop_image(image_url, label_str)
    rows = zip(file_names, unicodes, chars)
    with open(file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write the header only if the file doesn't exist
        if not file_exists:
            writer.writerow(["image_urls", "unicode", "char"])
        
        # Write the rows
        writer.writerows(rows)
        print(f"Sample {i+1} successed!")
        
log = pd.read_csv("data/reduce_classifier.csv")
unique_chars = log['char'].unique()
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(unique_chars)
mapping =  dict(zip(unique_chars, encoded_labels))
log['labels'] = log['char'].map(mapping)
log.to_csv("data/classifier.csv", index = False)
