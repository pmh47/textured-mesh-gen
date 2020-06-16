
import os
import sys
from itertools import islice, count
from collections import deque
import skvideo.io
import cv2

from crop_extraction_common import *
from bcs_common import *


step = 10
median_count = 25  # should be odd; number of stepped-frames to include in median (half either side of current frame)

crops_per_file = 256

score_threshold = 0.7
edge_distance = 20
min_box_area = 1500  # square raw-pixels

crop_border_fraction = 0.075  # per side

dataset_name = sys.argv[1]
masks_path = './bcs_masks'


def main():

    crops_path = bcs_preprocessed_path + '/dataset/' + dataset_name + '/crops_{}x{}_{}-per-file'.format(crop_width, crop_height, crops_per_file)
    if not os.path.exists(crops_path):
        os.makedirs(crops_path)

    occluder_mask = cv2.imread(masks_path + '/' + dataset_name + '_occluders.png')
    occluder_mask = cv2.resize(occluder_mask, (raw_width, raw_height), interpolation=cv2.INTER_NEAREST)
    occluder_mask = np.max(occluder_mask, axis=2)

    rotations = cv2.imread(masks_path + '/' + dataset_name + '_rotations.png')[:, :, 1]  # i.e. take just G channel, which is rotation
    rotations = cv2.resize(rotations, (raw_width, raw_height), interpolation=cv2.INTER_NEAREST)
    rotations = rotations.astype(np.float32) * 2. * np.pi / 255

    projection_matrix, view_rotation_matrix = load_calibration_matrices(dataset_name)

    detectron = DetectronWrapper(score_threshold)

    def stepped_indices_and_frames():

        reader = skvideo.io.vreader(bcs_raw_path + '/dataset/' + dataset_name + '/video.avi')
        buffer = deque()

        indices_and_frames = zip(count(), reader)
        for index, frame in islice(indices_and_frames, 0, None, step):
            buffer.append(frame[:, :, ::-1])  # convert BGR to RGB
            if len(buffer) == median_count:  # i.e. start considering frames only when we have enough to compute a median
                yield (
                    index - median_count // 2 * step,
                    buffer[median_count // 2],
                    lambda box: np.median(np.asarray([extract_crop(other_frame, raw_width, raw_height, crop_width, crop_border_fraction, *box[:-1])[0] for other_frame in buffer]), axis=0),
                )
                buffer.popleft()

    with ShardedRecordWriter(crops_path + '/{:04d}_l.tfrecords', crops_per_file) as left_writer:
        with ShardedRecordWriter(crops_path + '/{:04d}_r.tfrecords', crops_per_file) as right_writer:

            left_crop_count = right_crop_count = 0
            for frame_index, frame, get_median_crop in stepped_indices_and_frames():

                # Note that boxes are indexed by x1/y1/x2/y2/score

                class_to_boxes_and_masks = detectron(frame)

                def touches_edge(box, mask):
                    if box[0] < edge_distance or box[2] > frame.shape[1] - edge_distance:
                        return True
                    if box[1] < edge_distance or box[3] > frame.shape[0] - edge_distance:
                        return True
                    return False

                def touches_occluder(box, mask):
                    dilated_mask = cv2.dilate(mask, np.ones([3, 3]))
                    if np.count_nonzero(dilated_mask * occluder_mask) > 0:
                        return True
                    return False

                def touches_other_instance(box, mask):
                    dilated_mask = cv2.dilate(mask, np.ones([3, 3]))
                    for boxes_and_masks in class_to_boxes_and_masks.values():
                        for _, other_mask in boxes_and_masks:
                            if other_mask is mask:
                                continue
                            if np.count_nonzero(dilated_mask * other_mask) > 0:
                                return True
                    return False

                def is_large_enough(box, mask):
                    return box_area(box) > min_box_area

                car_boxes_and_masks = list(filter(
                    lambda box_and_mask:
                        not touches_edge(*box_and_mask) and not touches_occluder(*box_and_mask) and not touches_other_instance(*box_and_mask) and is_large_enough(*box_and_mask),
                    class_to_boxes_and_masks['car']
                ))

                for box, mask in car_boxes_and_masks:
                    assert crop_width == crop_height
                    crop, crop_raw_size = extract_crop(frame, raw_width, raw_height, crop_width, crop_border_fraction, *box[:-1])
                    cropped_mask, _ = extract_crop(mask, raw_width, raw_height, crop_width, crop_border_fraction, *box[:-1])
                    median = get_median_crop(box)
                    centre_x = int((box[0] + box[2]) // 2 + 0.5)
                    centre_y = int((box[1] + box[3]) // 2 + 0.5)
                    rotation = rotations[centre_y, centre_x]
                    offset_x = centre_x - raw_width / 2.
                    offset_y = raw_height / 2. - centre_y
                    crop_offset_matrix = [
                        [1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [-offset_x, -offset_y, 0., 1.],
                    ]
                    xy_ndc_matrix = np.diag([2. / crop_raw_size, 2. / crop_raw_size, 1., 1.])
                    crop_projection_matrix = np.dot(np.dot(projection_matrix, crop_offset_matrix), xy_ndc_matrix)
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'pixels': jpeg_feature(crop),
                                'background': jpeg_feature(median),
                                'mask': png_feature(cropped_mask),
                                'rotation': float32_feature(rotation),
                                'projection_matrix': float32_feature(crop_projection_matrix),
                                'view_rotation_matrix': float32_feature(view_rotation_matrix),
                                'dataset_name': string_feature(dataset_name)
                            }
                        )
                    )
                    if abs(rotation - np.pi) < 0.1:
                        writer = right_writer
                        right_crop_count += 1
                    else:
                        writer = left_writer
                        left_crop_count += 1
                    writer.write(example.SerializeToString())

                if len(car_boxes_and_masks) > 0:
                    print('{} left-lane / {} right-lane crops written at frame {}'.format(left_crop_count, right_crop_count, frame_index))


if __name__ == '__main__':
    main()

