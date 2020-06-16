
import numpy as np
import tensorflow as tf
import cv2

try:
    from caffe2.python import workspace
    from detectron.core.config import assert_and_infer_cfg
    from detectron.core.config import cfg
    from detectron.core.config import merge_cfg_from_file
    from detectron.utils.io import cache_url
    import detectron.core.test_engine as infer_engine
    import detectron.datasets.dummy_datasets as dummy_datasets
    import detectron.utils.c2 as c2_utils
    import pycocotools.mask as mask_util

    c2_utils.import_detectron_ops()

except:
    pass

config_path = './detectron/configs/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x.yaml'
weights_url = 'https://dl.fbaipublicfiles.com/detectron/36761843/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x.yaml.06_35_59.RZotkLKI/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl'


class ShardedRecordWriter(object):

    def __init__(self, path_format, examples_per_shard):
        self._path_format = path_format
        self._examples_per_shard = examples_per_shard
        self._shard_index = 0
        self._example_index_in_shard = 0
        self._new_file()

    def _new_file(self):
        if self._shard_index > 0:
            self._writer.close()
        self._writer = tf.python_io.TFRecordWriter(self._path_format.format(self._shard_index))
        self._shard_index += 1
        self._example_index_in_shard = 0

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        self.close()

    def write(self, serialised_example):
        if self._example_index_in_shard == self._examples_per_shard:
            self._new_file()
        self._writer.write(serialised_example)
        self._example_index_in_shard += 1

    def close(self):
        self._writer.close()


def float32_feature(value):
    value = np.asarray([value]).flatten()
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def jpeg_feature(image):
    value = cv2.imencode('.jpg', image)[1].tostring()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def png_feature(image):
    value = cv2.imencode('.png', image)[1].tostring()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


class DetectronWrapper(object):

    # Note that only one instance of this class should be instantiated per process!

    def __init__(self, score_threshold):

        self.score_threshold = score_threshold

        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])

        merge_cfg_from_file(config_path)
        cfg.NUM_GPUS = 1
        weights_path = cache_url(weights_url, cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg(cache_urls=False)

        assert not cfg.MODEL.RPN_ONLY, 'RPN models are not supported'
        assert not cfg.TEST.PRECOMPUTED_PROPOSALS, 'Models that require precomputed proposals are not supported'

        self.model = infer_engine.initialize_model_from_cfg(weights_path)
        self.dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    def __call__(self, frame):

        with c2_utils.NamedCudaScope(0):
            boxes_by_class, masks_by_class, _ = infer_engine.im_detect_all(self.model, frame, None)

        class_to_boxes_and_masks = {
            self.dummy_coco_dataset.classes[class_index]: [
                (box, mask_util.decode(mask))
                for box, mask in zip(boxes, masks)
                if box[4] > self.score_threshold
            ]
            for class_index, (boxes, masks) in enumerate(zip(boxes_by_class, masks_by_class))
        }

        return class_to_boxes_and_masks


def extract_crop(frame, raw_width, raw_height, crop_size, crop_border_fraction, x1, y1, x2, y2):

    # Crop is centred on the bbox, and equal in size to the larger size plus a bit
    centre_x = int((x1 + x2) // 2 + 0.5)
    centre_y = int((y1 + y2) // 2 + 0.5)
    width, height = x2 - x1, y2 - y1
    size = max(width, height)
    size += size * 2. * crop_border_fraction
    size = int(size + 0.5)

    # Add borders to the 'raw' image, so the crop doesn't go off an edge
    left_padding = -min(centre_x - size // 2, 0)
    top_padding = -min(centre_y - size // 2, 0)
    right_padding = max(centre_x + size // 2 - raw_width, 0)
    bottom_padding = max(centre_y + size // 2 - raw_height, 0)
    centre_x += left_padding
    centre_y += top_padding
    padded_frame = cv2.copyMakeBorder(frame, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_REPLICATE)

    raw_crop = padded_frame[centre_y - size // 2: centre_y + size // 2, centre_x - size // 2: centre_x + size // 2]
    crop = cv2.resize(raw_crop, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
    return crop, size


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

