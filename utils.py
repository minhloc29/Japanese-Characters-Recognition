import tensorflow as tf
import numpy as np

def get_mask(img, labels):
   
    img = tf.io.read_file(img)  # Read image file
    img = tf.image.decode_image(img, channels=3)  # Decode the image into a tensor
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # Convert the image to [0, 1] range
    
    img = tf.image.resize(img, (512, 512))  # Resize to a fixed size (optional) 
    img = img.numpy()  # Convert tensor to numpy array for mask processing
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
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    return mask #tensor: W x H x C

def load_image(img_url, img_size=(512, 512), expand_dim=False):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Read and preprocess the image
    img = tf.io.read_file(img_url)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, img_size)  # Resize to target size
    img = img / 255.0  # Scale to [0, 1]
    
    # Normalize using mean and std
    img = (img - mean) / std
    
    if expand_dim:
        img = tf.expand_dims(img, axis=0)  # Add batch dimension
    
    return img #Tensor: W x H x C


