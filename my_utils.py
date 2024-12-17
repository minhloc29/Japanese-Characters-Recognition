import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv, json
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import copy
from skimage import measure
from shapely.geometry import Polygon, Point
import time
import math
import functools
from tqdm.notebook import tqdm
from parameters import FONT, FONT_SIZE
from matplotlib import font_manager as fm
from losses_and_metrics import bce_dice_loss, dice_coef, iou_score

def visualize_training_data(image_fn,
                            labels,
                            width=3,
                            y_first=False, FONT_SIZE = 100, visualize_mode = False):
    
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
    if visualize_mode == False:
        return np.asarray(imsource)
    imsource.show()
                                
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
    
def load_image(img_url, img_size=(512, 512), expand_dim=False, get_info = False, get_original = False): #for data_loader
    #this fucntion can be used as input to model
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # Reshape to (1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    # Load the image using PIL
    img = Image.open(img_url).convert("RGB")
    h, w = img.size
    resize_img = img.resize(img_size)  
    resize_img = np.array(resize_img) / 255.0 
    # Normalize the image using mean and std
    resize_img = (resize_img - mean) / std 
    if expand_dim:
        resize_img = np.expand_dims(resize_img, axis=0)  # Add batch dimension
    if get_original:
        img = np.array(img)
    if get_info == True and get_original == False:
        return resize_img, w, h
    if get_info == False and get_original == True:
        return resize_img, img
    if get_info == True and get_original == True:
        return resize_img, w, h, img
    return resize_img #numpy: 512x512x3

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
    #mask: input after model.predict
    
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


def visualize_in_batches():
    jp_font = fm.FontProperties(fname="NotoSansJP-Bold.ttf")
    log = pd.read_csv("data/reduce_classifier.csv")
    sample = log.sample(16)
    image_urls, chars = sample['image_urls'].to_list(), sample['char'].to_list()
    titles = log[:16]['char'].to_list()
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize = (20, 20))
    ax = ax.flatten() 
    for i in range(16):
        image = cv2.imread(image_urls[i])[:, :, ::-1]
        ax[i].set_title(titles[i], fontproperties = jp_font)
        ax[i].imshow(image)
    plt.show()
    
def draw_rects(center_coords,
               bbox_cluster,
               o_img):
    count_skipping = 0
    img = copy.deepcopy(o_img)
    for cluster_index in range(len(center_coords))[1:]:
        char_pixel = (bbox_cluster == cluster_index).astype(np.float32)

        horizontal_indicies = np.where(np.any(char_pixel, axis=0))[0]
        vertical_indicies = np.where(np.any(char_pixel, axis=1))[0]
        if len(horizontal_indicies) > 0 and len(vertical_indicies) > 0:
            x_min, x_max = horizontal_indicies[[0, -1]]
            y_min, y_max = vertical_indicies[[0, -1]]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        else:
            print("Warning: Empty horizontal or vertical indices. Skipping this cluster.")
            count_skipping += 1

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
    print(f"Total skipping clusters: {count_skipping}")
    return img
  
def seg_prediction(image_url,
                   bbox_thres=0.01,
                   center_thres=0.02,
                   show_mode=False):
    
    model = tf.keras.models.load_model('Models/Segmentation/model.keras',
                                   custom_objects = {"dice_coef": dice_coef, 
                                                     "iou_score": iou_score,
                                                     "bce_dice_loss": bce_dice_loss})
    
    print(f"Input image url: {image_url}")
    o_img = load_image(image_url, expand_dim=True)
    # predict
    start = time.time()
    pred_mask = model.predict(o_img)
    print(">>> Inference time: {}'s".format(time.time() - start))
    pred_bbox, pred_center = pred_mask[0][:, :, 0], pred_mask[0][:, :, 1]
    pred_bbox = (pred_bbox > bbox_thres).astype(np.float32)
    pred_center = (pred_center > center_thres).astype(np.float32)
    assert pred_bbox.shape == pred_center.shape

    center_coords = get_centers(pred_center.astype(np.uint8))
    no_center_points = len(center_coords)
    print(">>> N.o center points: {}".format(no_center_points))
    if len(center_coords) == 0:
        print(">>> Non-text")
        plt.imshow(np.squeeze(o_img))
        return
    bbox_cluster = get_labels(center_coords, pred_bbox)

    plt_img = draw_rects(center_coords, bbox_cluster, np.squeeze(o_img))
    if show_mode == False:
        return center_coords, o_img[0], plt_img, pred_bbox, pred_center, bbox_cluster  # noqa
    wanna_show = [o_img[0], plt_img, pred_center, bbox_cluster]
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (20, 20))
    ax = ax.flatten()
    for i in range(4):
        ax[i].imshow(wanna_show[i])
    plt.show()    


def visual_pred_gt(model,
                   img_fp,
                   img_labels,
                   bbox_thres=0.01,
                   center_thres=0.02):

    # test_id = img_fp.split("/")[-1][:-4]
    # img_labels = df_train[df_train["image_id"].isin(
    #     [test_id])]["labels"].values[0]
    char_labels = np.array(img_labels.split(' ')).reshape(-1, 5)

    # visual gt
    img = visualize_training_data(img_fp, img_labels, width=5)
    img = np.array(img).copy()
    # visual pred
    oimg, oh, ow = load_image(img_fp, get_info=True)
    oimg = np.expand_dims(oimg, axis=0)

    start = time.time()
    pred_mask = model.predict(oimg)
    print(">>> Inference time: {}'s".format(time.time() - start))
    pred_bbox, pred_center = pred_mask[0][:, :, 0], pred_mask[0][:, :, 1]
    pred_bbox = (pred_bbox > bbox_thres).astype(np.float32)
    pred_center = (pred_center > center_thres).astype(np.float32)
    assert pred_bbox.shape == pred_center.shape

    center_coords = get_centers(pred_center.astype(np.uint8))
    no_center_points = len(center_coords)
    
    # polygon_contours = make_contours(pred_bbox)
    # filtered_contours = filter_polygons_points_intersection(polygon_contours, center_coords)  # noqa
    # pred_bbox = vis_pred_bbox_polygon(pred_bbox, filtered_contours)
    

    print(">>> no predicted center: {}".format(no_center_points))
    print(">>> Gt no center points: {}".format(len(char_labels)))
    if len(center_coords) == 0:
        print(">>> Non-text")
        return img

    y_ratio = oh / 512
    x_ratio = ow / 512
    print(y_ratio, x_ratio)
    # draw centers
    print(center_coords.shape)
    for y, x in center_coords:
        x = int(x * x_ratio)
        y = int(y * y_ratio)
        cv2.circle(img, (x, y), 3, (0, 255, 0), 5)

    if no_center_points > 0:
        bbox_cluster = get_labels(center_coords, pred_bbox)

        for cluster_index in range(len(center_coords))[1:]:
            char_pixel = (bbox_cluster == cluster_index).astype(np.float32)

            try:
                horizontal_indicies = np.where(np.any(char_pixel, axis=0))[0]
                vertical_indicies = np.where(np.any(char_pixel, axis=1))[0]
                x_min, x_max = horizontal_indicies[[0, -1]]
                y_min, y_max = vertical_indicies[[0, -1]]
            except IndexError:
                continue

            x = x_min
            y = y_min
            w = x_max - x_min
            h = y_max - y_min

            # resize to origin yx
            x = int(x   * x_ratio)
            w = int(w * x_ratio)
            y = int(y * y_ratio)
            h = int(h * y_ratio)
            print(x, y, w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)

    return img


def make_contours(masks, flatten=True):
    """
    flatten: follow by coco's api
    """
    if masks.ndim == 2:
        masks = np.expand_dims(masks, axis=-1)

    masks = masks.transpose((2, 0, 1))

    segment_objs = []
    for mask in masks:
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            contour = np.flip(contour, axis=1)
            if flatten:
                segmentation = contour.ravel().tolist()
            else:
                segmentation = contour.tolist()
            segment_objs.append(segmentation)

    return segment_objs

def filter_polygons_points_intersection(polygon_contours, center_coords):
    
    final_cons = []
    for con in polygon_contours:
        if(len(con) % 2 != 0):
            continue
        polygon = Polygon(zip(con[::2], con[1::2]))
        for center in center_coords:
            point = Point(center[1], center[0])
            if polygon.contains(point):
                final_cons.append(con)
                break

    return final_cons
    
def vis_pred_bbox_polygon(pred_bbox, cons):
    """
    pred_bbox: 1st mask
    cons: list contours return from `make_contours` method
    """
    mask_ = Image.new('1', (512, 512))
    mask_draw = ImageDraw.ImageDraw(mask_, '1')

    for contour in cons:
        mask_draw.polygon(contour, fill=1)

    mask_ = np.array(mask_).astype(np.uint8)
    return mask_ * 255



def get_result(image_url, bbox_thres = 0.01, center_thres = 0.02, FONT = FONT, FONT_SIZE = FONT_SIZE):
    with open("data/classifier_labels.json", 'r') as f:
        cls_labels = json.load(f)

    seg_model = tf.keras.models.load_model('models/segmentation/model.keras',
                                   custom_objects = {"dice_coef": dice_coef, 
                                                     "iou_score": iou_score,
                                                     "bce_dice_loss": bce_dice_loss})
    cls_model = tf.keras.models.load_model("models/classification/model.keras")
    
    font = ImageFont.truetype(
    FONT, FONT_SIZE, encoding="utf-8"
    )
    resize_img, oh, ow, oimg = load_image(image_url, get_info=True, get_original=True, expand_dim=True)
    seg_pred = seg_model.predict(resize_img)[0]
    pred_bboxs, pred_centers = seg_pred[:, :, 0], seg_pred[:, :, 1]
    pred_bboxs = (pred_bboxs > bbox_thres).astype(np.float32)
    pred_centers = (pred_centers > center_thres).astype(np.float32)
    polygon_contours = make_contours(pred_bboxs)
    center_coords = get_centers(pred_centers.astype(np.uint8))
    num_center_coords = len(center_coords)

    filtered_contours = filter_polygons_points_intersection(polygon_contours, center_coords)
    pred_bboxs = vis_pred_bbox_polygon(pred_bboxs, filtered_contours)
    final_bboxs = vis_pred_bbox(pred_bboxs, center_coords=center_coords, width=2)

    y_ratio = oh / 512
    x_ratio = ow / 512

    pil_img = Image.fromarray(oimg).convert('RGBA')
    char_canvas = Image.new('RGBA', pil_img.size)
    char_draw = ImageDraw.Draw(char_canvas)
    print(char_draw)
    print("Number of center points: {}".format(num_center_coords))

    if num_center_coords > 0:
            bbox_cluster = get_labels(center_coords, pred_bboxs)
            # ignore background hex color (=0)
            for cluster_index in tqdm(range(len(center_coords))[1:]):

                char_pixel = (bbox_cluster == cluster_index).astype(np.float32)

                try:
                    horizontal_indicies = np.where(np.any(char_pixel, axis=0))[0]
                    vertical_indicies = np.where(np.any(char_pixel, axis=1))[0]
                    x_min, x_max = horizontal_indicies[[0, -1]]
                    y_min, y_max = vertical_indicies[[0, -1]]
                except IndexError:
                    continue

                x = x_min
                y = y_min
                w = x_max - x_min
                h = y_max - y_min

                # convert to original coordinates
                x = int(x * x_ratio)
                w = int(w * x_ratio)
                y = int(y * y_ratio)
                h = int(h * y_ratio)

                # set offset to crop character
                offset = 5  # percentage
                y_diff = math.ceil(h * offset / 100)
                x_diff = math.ceil(w * offset / 100)

                # expand area
                y_from = y - y_diff
                y_to = y + h + y_diff
                x_from = x - x_diff
                x_to = x + w + x_diff
            
                y_from, y_to, x_from, x_to = \
                    list(map(functools.partial(np.maximum, 0),
                                [y_from, y_to, x_from, x_to]))

                try:
                    char_img = oimg[y_from:y_to, x_from:x_to]
                    mean = np.mean(char_img, axis=(0, 1, 2))
                    std = np.std(char_img, axis=(0, 1, 2))
                    char_img = (char_img - mean) / std
                    char_img = resize_padding(char_img, 64)
                    char_img = np.expand_dims(char_img, axis = 0)
                    pred_label = cls_model.predict(char_img)
                    pred_label = str(np.argmax(pred_label[0]))
                    pred_char = cls_labels[pred_label]
                    print(pred_char)
                    char_draw.text(
                        (x + w + FONT_SIZE / 4, y + h / 2 - FONT_SIZE),
                        pred_char, fill=(0, 0, 255, 255),
                        font=font
                    )
                except Exception as e:
                    print(e)
    char_img = Image.alpha_composite(pil_img, char_canvas)
    char_img = char_img.convert("RGB")
    char_img = np.asarray(char_img)
    return char_img
    # result = Image.fromarray(char_img)
    # result.show()
    # final_bbox = cv2.resize(final_bboxs, (origin_w, origin_h))
    # final_center = cv2.resize(final_centers, (origin_w, origin_h))
