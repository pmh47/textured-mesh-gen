
import cv2
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Distribution, Bernoulli, Normal, Laplace, Uniform, FULLY_REPARAMETERIZED
from tensorflow_probability.python.internal.distribution_util import gen_new_seed
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops


rv_all = [
    'SilhouettePyramid',
    'SsimPyramid'
]


def _gaussian_pyramid(pixels, levels=None, no_channels=False):

    # pixels is indexed by *, y, x, channel; dimensions * and channel are assumed to have static shape
    # If levels is None, then it is set automatically such that the smallest scale is 1x1
    # If no_channels is True, then neither x nor the result includes the trailing channel dimension

    if no_channels:
        pixels = pixels[..., np.newaxis]

    if levels is None:
        size = max(int(pixels.get_shape()[-2]), int(pixels.get_shape()[-3]))
        levels = int(math.ceil(math.log(size) / math.log(2))) + 1

    assert levels > 0  # includes the original scale

    kernel_sigma = 1.
    kernel_size = 3
    kernel_1d = cv2.getGaussianKernel(kernel_size, kernel_sigma)
    kernel = tf.constant(np.tile((kernel_1d * kernel_1d.T)[:, :, np.newaxis, np.newaxis], [1, 1, int(pixels.get_shape()[-1]), 1]), dtype=tf.float32)

    pyramid = [tf.reshape(pixels, [-1] + pixels.get_shape()[-3:].as_list())]
    for level in range(levels - 1):
        downsampled = tf.nn.depthwise_conv2d(pyramid[-1], kernel, [1, 2, 2, 1], 'SAME')
        pyramid.append(downsampled)

    # original_size = tf.cast(tf.size(pixels), tf.float32)
    # return [level * original_size / tf.cast(tf.size(level), tf.float32) for level in pyramid]
    result_with_channels = [tf.reshape(level, pixels.get_shape()[:-3].concatenate(level.get_shape()[-3:])) for level in pyramid]

    if no_channels:
        return [result_level[..., 0] for result_level in result_with_channels]
    else:
        return result_with_channels


class SilhouettePyramid(Distribution):

    # candidate is indexed by *, y, x, channel

    def __init__(self, candidate, levels=None, operation='iou', validate_args=False, allow_nan_stats=True, name='SilhouettePyramid'):
        with ops.name_scope(name, values=[candidate]) as ns:
            self._candidate = array_ops.identity(candidate, name='candidate')
            self._levels = levels
            self._candidate_pyramid = _gaussian_pyramid(self._candidate, self._levels, no_channels=True)
            self._operation = operation
            super(SilhouettePyramid, self).__init__(
                dtype=tf.float32,
                parameters={'candidate': candidate},
                reparameterization_type=FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                name=ns
            )

    def _log_prob(self, x):
        # The resulting density here will be indexed by *, i.e. we sum over x & y; channel is removed when computing the silhouettes
        # tf.summary.image('silhouette_pyramid/candidate_and_gt_raw', tf.concat([x, self._background], axis=1))
        # tf.summary.image('silhouette_pyramid/candidate_and_gt_sil', tf.concat([gt_silhouette, self._candidate], axis=1)[..., np.newaxis])
        def single_log_iou(a, b):
            intersection = a * b
            # iou = intersection / (a + b - intersection + 1.e-9)
            # return tf.log(tf.reduce_mean(iou, axis=[1, 2]) + 1.e-20)
            iou = tf.reduce_sum(intersection, axis=[1, 2]) / (tf.reduce_sum(a + b - intersection, axis=[1,2]) + 1.e0)
            return tf.log(iou / 3. + 1.e-20)
        if self._operation == 'iou':
            return single_log_iou(self._candidate, x) * 1.e2
        elif self._operation == 'iou_pyr':
            # ** or maybe use a max-pooling pyramid here?
            x_pyramid = _gaussian_pyramid(x, self._levels, no_channels=True)
            return sum([
                single_log_iou(x_level, candidate_level)  # indexed by *
                for x_level, candidate_level in zip(x_pyramid, self._candidate_pyramid)
            ]) / len(x_pyramid) * 8.e1
        elif self._operation == 'hinge':  # penalise candidate \ gt, i.e. encourage the candidate to stay inside the gt
            difference = tf.nn.relu(self._candidate - x)
            # ** could consider normalising by union (or gt-size) here too
            return tf.log(1. - tf.reduce_mean(difference, axis=[1, 2]) + 1.e-20) * 1.e3
        else:
            assert False

    def _sample_n(self, n, seed=None):
        assert n == 1
        return self._candidate[np.newaxis, :]  # ** not very random!

    def _mean(self):
        return self._candidate

    def _mode(self):
        return self._candidate


class SsimPyramid(Distribution):

    # candidate is indexed by *, y, x, channel

    def __init__(self, candidate, validate_args=False, allow_nan_stats=True, name='SsimPyramid'):
        with ops.name_scope(name, values=[candidate]) as ns:
            self._candidate = array_ops.identity(candidate, name='candidate')
            super(SsimPyramid, self).__init__(
                dtype=tf.float32,
                parameters={'candidate': candidate},
                reparameterization_type=FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                name=ns
            )

    def _log_prob(self, x):
        # The resulting density here will be indexed by *, i.e. we reduce over x, y, and channel
        weights = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
        if x.get_shape()[-2].value <= 140 or x.get_shape()[-3].value <= 140:
            weights = weights[:-1]
        return tf.log(tf.image.ssim_multiscale(x, self._candidate, 1., weights) + 1.e-6) * 90.

    def _sample_n(self, n, seed=None):
        assert n == 1
        return self._candidate[np.newaxis, :]  # ** not very random!

    def _mean(self):
        return self._candidate

    def _mode(self):
        return self._candidate

