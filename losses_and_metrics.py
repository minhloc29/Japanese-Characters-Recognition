import tensorflow as tf
from tf.keras.losses import CategoricalFocalCrossentropy 

def dice_coef(y_true, y_pred, threshold=0.5, epsilon=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]) > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + epsilon)
    return score

def iou_score(y_true, y_pred, threshold=0.5, epsilon=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]) > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    score = intersection / (union + epsilon)
    return score

# Loss Function
def dice_loss(y_true, y_pred, epsilon=1e-6):
    # Flatten the tensors
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    # Intersection and dice score
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + epsilon)
    return 1. - score

def bce_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = bce_loss(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

#Classifier
def focal_loss():
    return CategoricalFocalCrossentropy(
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
    reduction='sum_over_batch_size',
    name='categorical_focal_crossentropy'
)
