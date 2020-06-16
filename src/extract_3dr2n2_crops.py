
import os
import numpy as np
from collections import namedtuple

from crop_extraction_common import *
from camera_calibration import convert_camera_to_gl


dataset_path = '../data/3D-R2N2'
preprocessed_path = '../preprocessed-data/3D-R2N2'
synset = '02691156'  # aeroplane
# synset = '02958343'  # car
# synset = '03001627'  # chair
# synset = '04256520'  # sofa
split = 'train'  # 'train' or 'test'

images_per_mesh = 24  # the number in the original dataset
raw_size = 137  # width and height of images in the original dataset
train_fraction = 0.8  # the first this-fraction of meshes are used for training; remainder for test

crop_size = 128
crops_per_file = 256

distance_scale = 6.  # camera distances are multiplied by this amount; it should be set s.t. our anchor shape in physical units projects as a similar size to the shapenet renderings


Metadata = namedtuple('Metadata', ['azimuth', 'elevation', 'planar_rotation', 'distance', 'fov'])


def load_metadata(filename):

    with open(filename, 'rt') as f:
        lines = f.readlines()
    assert len(lines) == images_per_mesh
    return [
        Metadata(*map(float, line.strip().split(' ')))
        for line in lines
    ]


def angles_to_rotation_matrix(azimuth, elevation, theta):
    # The resulting matrix left-multiplies vectors
    from math import cos, sin
    azimuth = -azimuth * np.pi / 180. + np.pi / 2
    elevation *= np.pi / 180.
    theta *= np.pi / 180.
    Ry = np.float32([
        [cos(azimuth), 0., -sin(azimuth)],
        [0., 1., 0.],
        [sin(azimuth), 0.,  cos(azimuth)]
    ])
    Rx = np.float32([
        [1.,             0.,              0.],
        [0., cos(elevation), -sin(elevation)],
        [0., sin(elevation),  cos(elevation)]
    ])
    Rz = np.float32([
        [cos(theta), -sin(theta), 0.],
        [sin(theta),  cos(theta), 0.],
        [0.,           0.,        1.]
    ])
    R = np.dot(Rz, np.dot(Rx, Ry))
    return R


def get_matrices_from_metadata(metadata):

    from math import tan

    focal_length = 0.5 / tan((metadata.fov * 1. * np.pi / 180) / 2)  # doubling of metadata.fov is an assumption given what Choy's images look like!
    K = np.diag([focal_length, focal_length, 1.])
    R = angles_to_rotation_matrix(metadata.azimuth, metadata.elevation, metadata.planar_rotation)
    projection_matrix, view_rotation_matrix = convert_camera_to_gl(K, R, None, near=0.1, far=100.)  # also converts matrices from left-multiplying to right-multiplying

    T = np.float32([0., 0., metadata.distance * distance_scale])
    view_translation_matrix = np.eye(4)
    view_translation_matrix[3, :3] = -T
    view_matrix = np.dot(view_rotation_matrix, view_translation_matrix)

    return view_matrix.astype(np.float32), projection_matrix.astype(np.float32)


def get_path_and_ids_for_split(synset, split):
    input_path = '{}/ShapeNetRendering/{}/'.format(dataset_path, synset)
    all_ids = sorted(os.listdir(input_path))
    split_index = int(len(all_ids) * train_fraction)
    if split == 'train':
        split_ids = all_ids[:split_index]
    elif split == 'test':
        split_ids = all_ids[split_index:]
    else:
        raise RuntimeError('invalid split')
    return input_path, split_ids


def main():

    output_path = '{}/{}x{}_{}-per-file/{}/{}'.format(preprocessed_path, crop_size, crop_size, crops_per_file, synset, split)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_path = '{}/ShapeNetRendering/{}/'.format(dataset_path, synset)

    input_path, split_ids = get_path_and_ids_for_split(synset, split)

    assert crop_size <= raw_size
    top_left = (raw_size - crop_size) // 2

    with ShardedRecordWriter(output_path + '/{:04d}.tfrecords', crops_per_file) as writer:

        for mesh_index, mesh_id in enumerate(split_ids):

            print('{} / {}: {}'.format(mesh_index + 1, len(split_ids), mesh_id))

            metadata = load_metadata('{}/{}/rendering/rendering_metadata.txt'.format(input_path, mesh_id))

            for image_index in range(images_per_mesh):

                image_filename = '{}/{}/rendering/{:02d}.png'.format(input_path, mesh_id, image_index)

                image_and_alpha = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)
                assert image_and_alpha.shape[0] == image_and_alpha.shape[1] == raw_size
                assert image_and_alpha.shape[2] == 4

                image = image_and_alpha[:, :, :3]
                alpha = image_and_alpha[:, :, -1]

                image = image[top_left : top_left + crop_size, top_left : top_left + crop_size]
                alpha = alpha[top_left : top_left + crop_size, top_left : top_left + crop_size]

                view_matrix, projection_matrix = get_matrices_from_metadata(metadata[image_index])

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'pixels': jpeg_feature(image),
                            'background': jpeg_feature(np.zeros_like(image)),
                            'mask': png_feature((alpha > 0).astype(np.float32)),
                            'rotation': float32_feature(0.),
                            'projection_matrix': float32_feature(projection_matrix),
                            'view_rotation_matrix': float32_feature(view_matrix),  # note that it is in fact a complete view matrix in our case!
                            'dataset_name': string_feature('3D-R2N2_' + synset)
                        }))
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    main()

