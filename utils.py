import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv, json
import cv2
from sklearn.cluster import KMeans
import copy

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


def resize_padding(image_array, desire_size = 64):
    ratio = desire_size / max(image_array.shape)
    new_size = tuple([int(dim * ratio) for dim in image_array.shape[:2]])
    resize_image = cv2.resize(image_array, (new_size[1], new_size[0]))
    delta_w = desire_size - new_size[1]
    delta_h = desire_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # make padding
    color = [0, 0, 0]
    resize_image = cv2.copyMakeBorder(resize_image, top, bottom, left,
                                right, cv2.BORDER_CONSTANT, value=color)
    return resize_image

#segmentation visualization
def get_centers(mask):
    """find center points by using contour method

    :return: [(y1, x1), (y2, x2), ...]
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            cx, cy = cnt[0][0]
        cy = int(np.round(cy))
        cx = int(np.round(cx))
        centers.append([cy, cx])
    centers = np.array(centers)
    return centers 
# việc convert về np.uint8 là bắt buộc!

def get_labels(center_coords,
               pred_bbox):
    kmeans = KMeans(len(center_coords), init=center_coords)
    kmeans.fit(center_coords)  # noqa

    x, y = np.where(pred_bbox > 0)
    pred_cluster = kmeans.predict(list(zip(x, y)))

    pred_bbox_ = copy.deepcopy(pred_bbox)
    pred_bbox_[x, y] = pred_cluster

    return pred_bbox_

def vis_pred_bbox(pred_bbox, center_coords, image_url = None, width=6):
    

    bbox_cluster = get_labels(center_coords, pred_bbox)
    if image_url == None:
        image = np.zeros((512, 512))
    else:
        image = cv2.imread(image_url)[:, :, ::-1]
        image = cv2.resize(image, (512, 512))
    pil_img = Image.fromarray(image).convert('RGBA')
    bbox_canvas = Image.new('RGBA', pil_img.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas)
#     center_canvas = Image.new('RGBA', pil_img.size)
#     center_draw = ImageDraw.Draw(center_canvas)

    # exclude background index
    for cluster_index in range(len(center_coords))[1:]:
        char_pixel = (bbox_cluster == cluster_index).astype(np.float32)

        horizontal_indicies = np.where(np.any(char_pixel, axis=0))[0]
        vertical_indicies = np.where(np.any(char_pixel, axis=1))[0]
        x_min, x_max = horizontal_indicies[[0, -1]]
        y_min, y_max = vertical_indicies[[0, -1]]

        # draw polygon
        bbox_draw.rectangle(
            (x_min, y_min, x_max, y_max), fill=(255, 255, 255, 0),
            outline=(255, 0, 0, 255), width=width
        )
        # draw center

    res_img = Image.alpha_composite(pil_img, bbox_canvas)
    res_img = res_img.convert("RGB")
    res_img = np.asarray(res_img)

    # normalize image
    res_img = res_img / 255
    res_img = res_img.astype(np.float32)
    return res_img
