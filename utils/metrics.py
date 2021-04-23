import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np

class MIOU(tf.keras.metrics.Metric):
    def __init__(self, name='Mean_IOU', **kwargs):
        super(MIOU, self).__init__(name=name, **kwargs)
        self.mean_iou = self.add_weight(name='miou', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.reduce_sum(tf.cast(tp, self.dtype))
        fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        fp = tf.reduce_sum(tf.cast(fp, self.dtype))
        fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        fn = tf.reduce_sum(tf.cast(fn, self.dtype))
        dice_score = tp/(tp+fp+fn)
        self.mean_iou.assign_add(dice_score)
    def result(self):
        return self.mean_iou
    def reset_states(self):
        K.set_value(self.mean_iou, 0.0)

class PSNR(tf.keras.metrics.Metric):
    def __init__(self, name='psnr', **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name='value', initializer='zeros')
    def update_state(self, y_true, y_pred):
        self.psnr.assign_add(tf.image.psnr(y_true, y_pred, max_val=255.0))
    def result(self):
        return self.psnr
    def reset_state(self):
        K.set_value(self.psnr, 0.0)

class SSIM(tf.keras.metrics.Metric):
    def __init__(self, name='ssim', **kwargs):
        super(SSIM, self).__init__(name=name, **kwargs)
        self.ssim = self.add_weight(name='value', initializer='zeros')
    def update_state(self, y_true, y_pred):
        self.ssim.assign_add(tf.image.ssim(y_true, y_pred, max_val=255.0))
    def result(self):
        return self.ssim
    def reset_state(self):
        K.set_value(self.ssim, 0.0)