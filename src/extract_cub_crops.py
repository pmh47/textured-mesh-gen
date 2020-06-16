
import os
import numpy as np
import scipy.io

from crop_extraction_common import *


dataset_path = '../data/CUB-200-2011'
cmr_cachedir_path = '../data/cmr-cachedir'
preprocessed_path = '../preprocessed-data/CUB-200-2011'

split = 'train'

crop_size = 192
crops_per_file = 256
augmentation_factor = 10


def get_matrices_from_sfm(sfm_annotation, crop_centre_x, crop_centre_y, crop_size, crop_rotation_matrix):

    # In the original data...
    # x is the horizontal / across-the-wings / mirror-symmetric axis
    # y is the front-to-back / beak-to-tail axis
    # z is the dorsal-frontal axis
    # In our conventions...
    # x is length = beak-to-tail
    # y is vertical = dorsal-frontal
    # z = mirror-symmetric = across-the-wings
    canonicalisation_matrix = np.float32([
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.],
    ]).T

    rotation_matrix = sfm_annotation.rot
    scale = sfm_annotation.scale
    translation = sfm_annotation.trans

    rotation_matrix = rotation_matrix.T
    scale_matrix = np.diag([1., 1., -1.]) * scale

    view_matrix = np.concatenate([
        np.concatenate([
            np.matmul(np.matmul(canonicalisation_matrix, rotation_matrix), scale_matrix),
            np.zeros([3, 1])
        ], axis=1),
        [[translation[0], translation[1], -2.732 * scale, 1.]]
    ], axis=0)

    near = 10.
    far = 1200.

    projection_matrix = np.float32([
        [2. / crop_size, 0., 0., -(crop_centre_x * 2 - 2) / crop_size],  # the -2 offset accounts for 1-indexing in matlab
        [0., -2. / crop_size, 0., (crop_centre_y * 2 - 2) / crop_size],
        [0., 0., -2. / (far - near), -(far + near) / (far - near)],
        [0., 0., 0., 1.],
    ]).T
    crop_rotation_matrix_4x4 = np.eye(4)
    crop_rotation_matrix_4x4[:2, :2] = crop_rotation_matrix
    projection_matrix = np.matmul(projection_matrix, crop_rotation_matrix_4x4)

    return view_matrix, projection_matrix


def main():

    output_path = '{}/{}x{}_{}-per-file_{}x-jr-augm/{}'.format(preprocessed_path, crop_size, crop_size, crops_per_file, augmentation_factor, split)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_infos = scipy.io.loadmat(cmr_cachedir_path + '/cub/data/{}_cub_cleaned.mat'.format(split), squeeze_me=True, struct_as_record=False)['images']
    sfm_annotations = scipy.io.loadmat(cmr_cachedir_path + '/cub/sfm/anno_{}.mat'.format(split), squeeze_me=True, struct_as_record=False)['sfm_anno']

    image_base_path = '{}/CUB_200_2011/images'.format(dataset_path)

    with ShardedRecordWriter(output_path + '/{:04d}.tfrecords', crops_per_file) as writer:

        for image_index, (image_info, sfm_annotation) in enumerate(zip(image_infos, sfm_annotations)):

            print('{} / {}: {}'.format(image_index + 1, len(image_infos), image_info.rel_path))

            image_filename = os.path.join(image_base_path, image_info.rel_path)

            image = cv2.imread(image_filename)
            assert image.shape[-1] == 3

            for _ in range(augmentation_factor):

                crop_jitter = np.random.uniform(-0.05, 0.05, size=[4])
                bbox = np.float32([image_info.bbox.x1, image_info.bbox.y1, image_info.bbox.x2, image_info.bbox.y2])
                raw_crop_width = bbox[2] - bbox[0]
                raw_crop_height = bbox[3] - bbox[1]
                bbox += crop_jitter * [raw_crop_width, raw_crop_height, raw_crop_width, raw_crop_height]
                bbox = (bbox + 0.5).astype(np.int32)
                crop_centre_x = (bbox[0] + bbox[2]) / 2
                crop_centre_y = (bbox[1] + bbox[3]) / 2

                crop_rotation = np.random.uniform(-15., 15.)  # degrees
                rotation_matrix = cv2.getRotationMatrix2D((crop_centre_x, crop_centre_y), crop_rotation, 1.0)
                rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
                rotated_mask = cv2.warpAffine(image_info.mask, rotation_matrix, image.shape[1::-1], borderMode=cv2.BORDER_REFLECT)

                cropped_image, crop_original_size = extract_crop(rotated_image, image.shape[1], image.shape[0], crop_size, 0.05, *bbox)
                cropped_mask, _ = extract_crop(rotated_mask, image.shape[1], image.shape[0], crop_size, 0.05, *bbox)

                view_matrix, projection_matrix = get_matrices_from_sfm(
                    sfm_annotation,
                    crop_centre_x, crop_centre_y,
                    crop_original_size,
                    rotation_matrix[:, :2]
                )

                background = cv2.inpaint(cropped_image, cv2.dilate(cropped_mask, np.ones([3, 3]), iterations=2), 5, cv2.INPAINT_TELEA)

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'pixels': jpeg_feature(cropped_image),
                            'background': jpeg_feature(background),
                            'mask': png_feature(cropped_mask.astype(np.float32)),
                            'rotation': float32_feature(0.),
                            'projection_matrix': float32_feature(projection_matrix),
                            'view_rotation_matrix': float32_feature(view_matrix),  # note that it is in fact a complete view matrix in our case!
                            'dataset_name': string_feature('CUB-200-2011')
                        }))
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    main()

