
import os
import numpy as np
import scipy.io

from crop_extraction_common import *


dataset_path = '../data/HSP'
preprocessed_path = '../preprocessed-data/HSP'
# synset = '02691156'  # aeroplane
# synset = '02958343'  # car
synset = '03001627'  # chair
# synset = '04256520'  # sofa
split = 'test'  # 'train' or 'val' or 'test'

images_per_mesh = 19  # the number in the original dataset
raw_size = 224  # width and height of images in the original dataset

crop_size = 192
crops_per_file = 256

base_scale = 0.3  # this results in a similar mesh scaling to the 3D-R2N2 data


def get_matrices(camera_info):

    quat = camera_info['quat']
    pos = camera_info['pos']
    K = camera_info['K']
    extrinsic = camera_info['extrinsic']

    # In our conventions...
    # x is length / front-to-back
    # y is vertical
    # z = mirror-symmetric / left-to-right
    canonicalisation_matrix = np.float32([
        [base_scale, 0., 0., 0.],
        [0., 0., base_scale, 0.],
        [0., -base_scale, 0., 0.],
        [0., 0., 0., 1.],
    ]).T

    near = 0.1
    far = 10.
    perspective_matrix = np.asarray([
        [-K[0, 0], -K[0, 1], 0, 0],  # ** why is the negation here necessary?
        [K[1, 0], K[1, 1], 0, 0],
        [0, 0, near + far, near * far],
        [0, 0, -K[2, 2], 0]
    ])
    ndc_matrix = np.asarray([
        [2. / raw_size, 0, 0, 0],
        [0, 2. / raw_size, 0, 0],
        [0, 0, -2. / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1]
    ])
    projection_matrix = np.dot(ndc_matrix, perspective_matrix).T

    view_matrix = extrinsic.T
    view_matrix[3, 2] = -2.
    view_matrix = np.matmul(canonicalisation_matrix, view_matrix)

    return view_matrix, projection_matrix


def get_ids_for_split(synset, split):

    with open('{}/shapenet_data/shapenet/{}_{}.txt'.format(dataset_path, split, synset)) as f:
        split_ids = [
            line[:-5]
            for line in f.readlines()
        ]
    return split_ids


def main():

    output_path = '{}/{}x{}_{}-per-file/{}/{}'.format(preprocessed_path, crop_size, crop_size, crops_per_file, synset, split)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_path = '{}/shapenet_data/shapenet/blenderRenderPreprocess/{}/'.format(dataset_path, synset)

    split_ids = get_ids_for_split(synset, split)

    with ShardedRecordWriter(output_path + '/{:04d}.tfrecords', crops_per_file) as writer:

        for mesh_index, mesh_id in enumerate(split_ids):

            print('{} / {}: {}'.format(mesh_index + 1, len(split_ids), mesh_id))

            for image_index in range(1, images_per_mesh + 1):

                image_filename = '{}/{}/render_{}.png'.format(input_path, mesh_id, image_index)

                image_and_alpha = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)
                assert image_and_alpha.shape[0] == image_and_alpha.shape[1] == raw_size
                assert image_and_alpha.shape[2] == 4

                image_and_alpha = cv2.resize(image_and_alpha, (crop_size, crop_size))

                image = image_and_alpha[:, :, :3]
                alpha = image_and_alpha[:, :, -1]

                view_matrix, projection_matrix = get_matrices(scipy.io.loadmat('{}/{}/camera_{}.mat'.format(input_path, mesh_id, image_index), squeeze_me=True))

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'pixels': jpeg_feature(image),
                            'background': jpeg_feature(np.zeros_like(image)),
                            'mask': png_feature(alpha.astype(np.float32) / 255.),
                            'rotation': float32_feature(0.),
                            'projection_matrix': float32_feature(projection_matrix),
                            'view_rotation_matrix': float32_feature(view_matrix),  # note that it is in fact a complete view matrix in our case!
                            'dataset_name': string_feature('HSP_' + synset)
                        }))
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    main()

