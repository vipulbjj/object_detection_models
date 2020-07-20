import keras.backend as K
import tensorflow as tf
from keras.losses import mean_absolute_error
import numpy as np
import cv2
from PIL import Image

def read_image_bgr(path):

    image = np.ascontiguousarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1]

def focal_loss(heatmap_pred, heatmap_true):
    pos_mask = tf.cast(tf.equal(heatmap_true, 1), tf.float32)
    neg_mask = tf.cast(tf.less(heatmap_true, 1), tf.float32)# Returns the truth value of (x < y) element-wise.
    neg_weights = tf.pow(1 - heatmap_true, 4)

    pos_loss = -tf.log(tf.clip_by_value(heatmap_pred, 1e-4, 1. - 1e-4)) * tf.pow(1 - heatmap_pred, 2) * pos_mask
    neg_loss = -tf.log(tf.clip_by_value(1 - heatmap_pred, 1e-4, 1. - 1e-4)) * tf.pow(heatmap_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss


def reg_l1_loss(y_pred, y_true, indices, mask):
    b = tf.shape(y_pred)[0]
    k = tf.shape(indices)[1]
    c = tf.shape(y_pred)[-1]
    y_pred = tf.reshape(y_pred, (b, -1, c))
    indices = tf.cast(indices, tf.int32)
    y_pred = tf.gather(y_pred, indices, batch_dims=1)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
    return reg_loss

def centernet_loss(args):
    #This includes both the above losses
    
    heatmap_pred, width_height_pred, offset_pred, heatmap_true, width_height_true, offset_true, offset_mask, indices = args
    heatmap_loss = focal_loss(heatmap_pred, heatmap_true)
    width_height_loss = 0.1 * reg_l1_loss(width_height_pred, width_height_true, indices, offset_mask)
    offset_loss = reg_l1_loss(offset_pred, offset_true, indices, offset_mask)
    total_loss = heatmap_loss + width_height_loss + offset_loss
    return total_loss

def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, (kernel, kernel), strides=1, padding='SAME')
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat

def topk(heatmap, max_objects=100):
    heatmap = nms(heatmap)# (b, h * w * c)
    b, h, w, c = tf.shape(heatmap)[0], tf.shape(heatmap)[1], tf.shape(heatmap)[2], tf.shape(heatmap)[3]
    hm = tf.reshape(heatmap, (b, -1))# (b, k), (b, k)
    scores, indices = tf.nn.top_k(heatmap, k=1)#changed k from max_objects. need to check
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys

def evaluate_batch_item(batch_item_detections, num_classes, max_objects_per_class=20, max_objects=100,
                        iou_threshold=0.5, score_threshold=0.1):
    batch_item_detections = tf.boolean_mask(batch_item_detections,
                                            tf.greater(batch_item_detections[:, 4], score_threshold))
    detections_per_class = []
    for cls_id in range(num_classes):
        class_detections = tf.boolean_mask(batch_item_detections, tf.equal(batch_item_detections[:, 5], cls_id))
        nms_keep_indices = tf.image.non_max_suppression(class_detections[:, :4],
                                                        class_detections[:, 4],
                                                        max_objects_per_class,
                                                        iou_threshold=iou_threshold)
        class_detections = K.gather(class_detections, nms_keep_indices)
        detections_per_class.append(class_detections)

    batch_item_detections = K.concatenate(detections_per_class, axis=0)

    def filter():
        nonlocal batch_item_detections
        _, indices = tf.nn.top_k(batch_item_detections[:, 4], k=max_objects)
        batch_item_detections_ = tf.gather(batch_item_detections, indices)
        return batch_item_detections_

    def pad():
        nonlocal batch_item_detections
        batch_item_num_detections = tf.shape(batch_item_detections)[0]
        batch_item_num_pad = tf.maximum(max_objects - batch_item_num_detections, 0)
        batch_item_detections_ = tf.pad(tensor=batch_item_detections,
                                        paddings=[
                                            [0, batch_item_num_pad],
                                            [0, 0]],
                                        mode='CONSTANT',
                                        constant_values=0.0)
        return batch_item_detections_

    batch_item_detections = tf.cond(tf.shape(batch_item_detections)[0] >= 100,
                                    filter,
                                    pad)
    return batch_item_detections
    
def decode(heatmap, width_height, offset, max_objects=100, nms=True, flip_test=False, num_classes=20, score_threshold=0.1):
    if flip_test:
        heatmap = (heatmap[0:1] + heatmap[1:2, :, ::-1]) / 2
        width_height = (width_height[0:1] + width_height[1:2, :, ::-1]) / 2
        offset = offset[0:1]
    scores, indices, class_ids, xs, ys = topk(heatmap, max_objects=max_objects)
    b = tf.shape(heatmap)[0]# (b, h * w, 2)
    offset = tf.reshape(offset, (b, -1, tf.shape(offset)[-1]))# (b, h * w, 2)
    width_height = tf.reshape(width_height, (b, -1, tf.shape(width_height)[-1]))# (b, k, 2)
    topk_offset = tf.gather(offset, indices, batch_dims=1)# (b, k, 2)
    topk_width_height = tf.cast(tf.gather(width_height, indices, batch_dims=1), tf.float32)
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_offset[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_offset[..., 1:2]
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = topk_cx - topk_width_height[..., 0:1] / 2
    topk_x2 = topk_cx + topk_width_height[..., 0:1] / 2
    topk_y1 = topk_cy - topk_width_height[..., 1:2] / 2
    topk_y2 = topk_cy + topk_width_height[..., 1:2] / 2# (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    if nms:
        detections = tf.map_fn(lambda x: evaluate_batch_item(x[0],
                                                             num_classes=num_classes,
                                                             score_threshold=score_threshold),
                               elems=[detections],
                               dtype=tf.float32)
    return detections
    
def get_affine_transform(center,
                         scale,
                         output_size,
                         rot=0.,
                         inv=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list) and not isinstance(scale, tuple):
        scale = np.array([scale, scale], dtype=np.float32)

    if not isinstance(output_size, np.ndarray) and not isinstance(output_size, list) and not isinstance(output_size,
                                                                                                        tuple):
        output_size = np.array([output_size, output_size], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    src_h = scale_tmp[1]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_h * -0.5], rot_rad)
    dst_dir = np.array([0, dst_h * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def draw_gaussian(heatmap, center, radius_h, radius_w, k=1):
    diameter_h = 2 * radius_h + 1
    diameter_w = 2 * radius_w + 1
    gaussian = gaussian2D((diameter_h, diameter_w), sigma_w=diameter_w / 6, sigma_h=diameter_h / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_w), min(width - x, radius_w + 1)
    top, bottom = min(y, radius_h), min(height - y, radius_h + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius_h - top:radius_h + bottom, radius_w - left:radius_w + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_gaussian_2(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D_2((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def gaussian2D(shape, sigma_w=1, sigma_h=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-((x * x) / (2 * sigma_w * sigma_w) + (y * y) / (2 * sigma_h * sigma_h)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    det_h, det_w = det_size
    rh = 0.1155 * det_h
    rw = 0.1155 * det_w
    return rh, rw


def gaussian_radius_2(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)
    
def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def gaussian2D_2(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h
if __name__ == '__main__':
    print("---------------------------------------")