
import os
import cv2
import json
import pickle
import numpy as np
from datetime import datetime, date
import tensorflow as tf
from tensorflow.contrib import slim
import meshzoo
from tqdm import tqdm
import argparse

import dirt
import dirt.matrices
import dirt.lighting

from mesh_utils import *
from random_variables import Normal, Uniform, Bernoulli, SsimPyramid, SilhouettePyramid
from integrated_klqp import noncopying_integrated_reparam_klqp, GenerativeMode
import camera_calibration
from hyperparams import Hyperparams


hyper = Hyperparams()


output_path = '../output'

random_seed = hyper(1, 'seed', int)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)


write_images_frequency = 50  # iterations

write_eval_images_frequency = 2500
eval_images_count = 12800  # used for FID / etc.
eval_meshes_count = 2560  # used for COV-CD / etc.

val_fraction = 0.15
images_per_batch = 64


dataset = hyper('shapenet', 'dataset', str)
with_gt_masks = hyper(1, 'with-gt-masks', int)

shape_model = hyper('VAE', 'shape-model', str)
assert shape_model in {'VAE', 'FA', 'VAE-seq-att', 'VAE-seq-att-pushing'}


if dataset == 'bcs':

    from bcs_common import bcs_preprocessed_path, crop_width, crop_height, crops_per_file, default_camera_height

    bcs_sequences = ['session1_left', 'session4_left', 'session4_right', 'session5_left', 'session5_center', 'session5_right', 'session6_left', 'session6_right']
    data_paths = [bcs_preprocessed_path + '/dataset/' + sequence_name + '/crops_{}x{}_{}-per-file'.format(crop_width, crop_height, crops_per_file) for sequence_name in bcs_sequences]
    dataset_string = 'bcs-' + ','.join(bcs_sequences) + '_{}x{}_{}pf_lr_ws_2019-03'.format(crop_width, crop_height, crops_per_file)

    epochs = 50

    if shape_model == 'VAE-seq-att-pushing':
        base_mesh = 'cube'
        subdivision_count = 7
    else:
        base_mesh = 'sphere'
        subdivision_count = 4
    sphere_scale = 1.5

    anchor_scaling = [2.35, 1.2, 1.4]  # used by generative; implies cars point along x-axis
    sun_direction = np.float32([-0.05, -1, -0.05]) / (1 + 0.05**2 + 0.05**2) ** 0.5
    shadow_scaling = [0.2, 0.2, 0.2]  # background r/g/b values are multiplied by this when in shadow

    shadow_map_width = shadow_map_height = 384
    shadow_stride = 4  # screen-space pixels are downsampled by this factor before projection into shadow map

    flip_augmentation = True

    if with_gt_masks:
        l2_laplacian_strength = 1.e2; tv_l1_strength = 0.; crease_strength = 1.e2; l2_grad_strength = 0.; equilat_strength = 0.
    else:
        l2_laplacian_strength = 5.e1; tv_l1_strength = 0.; crease_strength = 1.e1; l2_grad_strength = 0.; equilat_strength = 0.

    background_embedding_dimensionality = None if with_gt_masks else 16

elif dataset == 'shapenet':

    shapenet_synset = hyper('02958343', 'synset', str)  # 02691156 (aeroplane), 02958343 (car), 03001627 (chair), 04256520 (sofa)
    shapenet_source = hyper('HSP' if shapenet_synset in {'02691156', '03001627'} else '3D-R2N2', 'shapenet-src', str)  # '3D-R2N2' or 'HSP'

    if shapenet_source == '3D-R2N2':
        crop_width = crop_height = 128
    elif shapenet_source == 'HSP':
        crop_width = crop_height = 192
    else:
        assert False

    shapenet_preprocessed_path = '../preprocessed-data'
    shapenet_suffix = '{}x{}_256-per-file'.format(crop_width, crop_height)

    dataset_string = 'shapenet-{}_{}_{}_colour-bg'.format(shapenet_source, shapenet_suffix, shapenet_synset)

    epochs = 100

    # Note z is the horizontal (mirror-symmetric) axis, y is vertical, and x is lengthwise/depth
    if shapenet_synset == '02691156':  # aeroplane
        anchor_scaling = [1.2, 0.8, 2.]
        flip_augmentation = True
        l2_laplacian_strength = 0.; tv_l1_strength = 1.e0; crease_strength = 1.e0; l2_grad_strength = 0.; equilat_strength = 0.
        subdivision_count = 7 if shape_model == 'VAE-seq-att-pushing' else 3
    elif shapenet_synset == '02958343':  # car
        anchor_scaling = [2.35, 1.2, 1.4]
        flip_augmentation = True
        if with_gt_masks:
            l2_laplacian_strength = 1.e0; tv_l1_strength = 0.; crease_strength = 1.e1; l2_grad_strength = 0.; equilat_strength = 0.
        else:
            l2_laplacian_strength = 1.e1; tv_l1_strength = 0.; crease_strength = 1.e1; l2_grad_strength = 0.; equilat_strength = 0.
        subdivision_count = 7 if shape_model == 'VAE-seq-att-pushing' else 4
    elif shapenet_synset == '03001627':  # chair
        anchor_scaling = [1.2, 1.6, 0.8]
        flip_augmentation = True
        if shape_model == 'VAE-seq-att-pushing':
            l2_laplacian_strength = 1.e0; tv_l1_strength = 0.; crease_strength = 1.e0; l2_grad_strength = 0.; equilat_strength = 0.
        else:
            l2_laplacian_strength = 0.; tv_l1_strength = 1.e1; crease_strength = 0.; l2_grad_strength = 0.; equilat_strength = 0.
        subdivision_count = 9 if shape_model == 'VAE-seq-att-pushing' else 3
    elif shapenet_synset == '04256520':  # sofa
        anchor_scaling = [1.2, 1.2, 2.5]
        flip_augmentation = False  # due to right-angle designs
        if shape_model == 'VAE-seq-att-pushing':
            l2_laplacian_strength = 1.e0; tv_l1_strength = 1.e1; crease_strength = 1.e0; l2_grad_strength = 0.; equilat_strength = 0.
        else:
            l2_laplacian_strength = 0.; tv_l1_strength = 0.; crease_strength = 1.e0; l2_grad_strength = 0.; equilat_strength = 0.
        subdivision_count = 6 if shape_model == 'VAE-seq-att-pushing' else 3
    else:
        assert False

    sun_direction = tf.linalg.l2_normalize([-0.05, -1, -0.05])
    base_mesh = 'cube' if shape_model == 'VAE-seq-att-pushing' else 'sphere'
    sphere_scale = 1.

    background_embedding_dimensionality = None if with_gt_masks else 32

elif dataset == 'cub':

    crop_width = crop_height = 192

    cub_preprocessed_path = '../preprocessed-data/CUB-200-2011'
    cub_suffix = '{}x{}_256-per-file_10x-jr-augm'.format(crop_width, crop_height)

    dataset_string = 'cub_{}'.format(cub_suffix)

    epochs = 100

    anchor_scaling = [1.4, 0.8, 1.4]  # x is beak-to-tail; y is dorsal-frontal; z is across-the-wings
    flip_augmentation = True

    sun_direction = tf.linalg.l2_normalize([-0.05, -1, -0.05])
    if shape_model == 'VAE-seq-att-pushing':
        base_mesh = 'cube'
        subdivision_count = 7
    else:
        base_mesh = 'sphere'
        subdivision_count = 4
    sphere_scale = 1.

    l2_laplacian_strength = 1e1; tv_l1_strength = 0.; crease_strength = 1e0; l2_grad_strength = 0.; equilat_strength = 0.

    background_embedding_dimensionality = None if with_gt_masks else 16

else:

    assert False


recon_image_count = 8
turntable_orientations = 6
turntable_camera_elevation = 20. * np.pi / 180.
turntable_camera_distance = 8.5


sequential_attentive_steps = 4

joint_embedding_dimensionality = None
shape_embedding_dimensionality = 32
colour_embedding_dimensionality = 128
colour_decoder_hidden_dimensionality = 192


def decode_serialised_example(serialised_example):

    features = tf.parse_single_example(
        serialised_example,
        features={
            'pixels': tf.FixedLenFeature([], tf.string),
            'background': tf.FixedLenFeature([], tf.string),
            'mask': tf.FixedLenFeature([], tf.string),
            'rotation': tf.FixedLenFeature([], tf.float32),
            'projection_matrix': tf.FixedLenFeature([4, 4], tf.float32),
            'view_rotation_matrix': tf.FixedLenFeature([4, 4], tf.float32),  # note that for shapenet and cub, this is in fact the complete view matrix
            'dataset_name': tf.FixedLenFeature([], tf.string),
        })

    pixels = tf.image.decode_jpeg(features['pixels'])
    pixels.set_shape([crop_height, crop_width, 3])

    background = tf.image.decode_jpeg(features['background'])
    background.set_shape([crop_height, crop_width, 3])

    mask = tf.image.decode_png(features['mask'])[:, :, 0]
    mask.set_shape([crop_height, crop_width])

    return pixels, background, mask, features['rotation'], features['projection_matrix'], features['view_rotation_matrix'], features['dataset_name']


class BcsGroundTruth(object):

    @staticmethod
    def _get_dataset_for_split(left_filenames, right_filenames, repeats):

        np.random.shuffle(left_filenames)  # note that this must occur *after* the train/val split
        np.random.shuffle(right_filenames)

        left_dataset = tf.data.TFRecordDataset(left_filenames)
        right_dataset = tf.data.TFRecordDataset(right_filenames)

        assert len(left_filenames) > 0 and len(right_filenames) > 0
        left_dataset = left_dataset.repeat(repeats if len(right_filenames) < len(left_filenames) else None)
        right_dataset = right_dataset.repeat(repeats if len(left_filenames) < len(right_filenames) else None)
        dataset = tf.data.Dataset.zip((left_dataset, right_dataset)).flat_map(
            lambda left, right:
                tf.data.Dataset.from_tensors(left).concatenate(tf.data.Dataset.from_tensors(right))
        )

        dataset = dataset.map(decode_serialised_example)
        dataset = dataset.shuffle(2000 + 3 * images_per_batch)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(images_per_batch))

        return dataset

    def __init__(self, is_training):

        left_train_filenames = []
        right_train_filenames = []
        left_val_filenames = []
        right_val_filenames = []
        for data_path in data_paths:
            filenames_for_path = list(map(lambda filename: data_path + '/' + filename, sorted(os.listdir(data_path))))  # sorting ensures train/val split is at a single point in time
            left_filenames_for_path = list(filter(lambda filename: filename.endswith('_l.tfrecords'), filenames_for_path))
            right_filenames_for_path = list(filter(lambda filename: filename.endswith('_r.tfrecords'), filenames_for_path))
            assert len(left_filenames_for_path) + len(right_filenames_for_path) == len(filenames_for_path)

            left_first_val_index = int(len(left_filenames_for_path) * (1. - val_fraction) + 0.5)
            assert left_first_val_index > 0
            left_train_filenames.extend(left_filenames_for_path[:left_first_val_index])
            left_val_filenames.extend(left_filenames_for_path[left_first_val_index:])

            right_first_val_index = int(len(right_filenames_for_path) * (1. - val_fraction) + 0.5)
            assert right_first_val_index > 0
            right_train_filenames.extend(right_filenames_for_path[:right_first_val_index])
            right_val_filenames.extend(right_filenames_for_path[right_first_val_index:])

        print('using {} folders'.format(len(data_paths)))
        print('training (left): {} files (slightly under {} crops)'.format(len(left_train_filenames), len(left_train_filenames) * crops_per_file))
        print('training (right): {} files (slightly under {} crops)'.format(len(right_train_filenames), len(right_train_filenames) * crops_per_file))
        print('validation (left): {} files (slightly under {} crops)'.format(len(left_val_filenames), len(left_val_filenames) * crops_per_file))
        print('validation (right): {} files (slightly under {} crops)'.format(len(right_val_filenames), len(right_val_filenames) * crops_per_file))

        train_dataset = self._get_dataset_for_split(left_train_filenames, right_train_filenames, epochs)
        val_dataset = self._get_dataset_for_split(left_val_filenames, right_val_filenames, None)

        self.pixels, self.background, self.mask, self.rotation, self.projection_matrix, self.view_rotation_matrix, dataset_name = tf.cond(
            is_training,
            lambda: train_dataset.make_one_shot_iterator().get_next(),
            lambda: val_dataset.make_one_shot_iterator().get_next()
        )

        self.pixels = tf.cast(self.pixels, tf.float32) / 255.
        self.background = tf.cast(self.background, tf.float32) / 255.

        self.dataset_count = len(bcs_sequences)
        dataset_table = tf.contrib.lookup.index_table_from_tensor(tf.constant(bcs_sequences))
        self.dataset_index = dataset_table.lookup(dataset_name)
        assert_valid_name = tf.Assert(tf.reduce_all(self.dataset_index >= 0), [dataset_name])
        with tf.control_dependencies([assert_valid_name]):
            self.dataset_index = tf.identity(self.dataset_index)


def get_dataset_from_folders(*folders):

    filenames = sum([
            [os.path.join(folder, filename) for filename in sorted(os.listdir(folder))]
            for folder in folders
    ], [])

    np.random.shuffle(filenames)
    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.repeat(epochs)
    dataset = dataset.map(decode_serialised_example)

    dataset = dataset.shuffle(2000 + 3 * images_per_batch)
    dataset = dataset.batch(images_per_batch, drop_remainder=True)

    return dataset


class ShapenetGroundTruth(object):

    @staticmethod
    def _get_dataset_for_splits(splits):
        return get_dataset_from_folders(*[os.path.join(shapenet_preprocessed_path, shapenet_source, shapenet_suffix, shapenet_synset, split) for split in splits])

    def __init__(self, is_training):

        if shapenet_source == '3D-R2N2':
            train_dataset = self._get_dataset_for_splits(['train'])
        elif shapenet_source == 'HSP':
            train_dataset = self._get_dataset_for_splits(['train', 'val'])
        else:
            assert False

        test_dataset = self._get_dataset_for_splits(['test'])

        self.pixels, self.background, self.mask, self.rotation, self.projection_matrix, self.view_rotation_matrix, dataset_name = tf.cond(
            is_training,
            lambda: train_dataset.make_one_shot_iterator().get_next(),
            lambda: test_dataset.make_one_shot_iterator().get_next()
        )

        background_colour = tf.random_uniform([images_per_batch, 3])
        float_mask = tf.cast(self.mask, tf.float32)[..., None]
        self.background = tf.ones([images_per_batch , crop_height, crop_width, 3]) * background_colour[:, None, None, :]
        self.pixels = tf.cast(self.pixels, tf.float32) / 255. * float_mask + self.background * (1. - float_mask)


class CubGroundTruth(object):

    @staticmethod
    def _get_dataset_for_splits(splits):

        return get_dataset_from_folders(*[os.path.join(cub_preprocessed_path, cub_suffix, split) for split in splits])

    def __init__(self, is_training):

        train_dataset = self._get_dataset_for_splits(['train', 'val'])
        test_dataset = self._get_dataset_for_splits(['test'])

        self.pixels, self.background, self.mask, self.rotation, self.projection_matrix, self.view_rotation_matrix, dataset_name = tf.cond(
            is_training,
            lambda: train_dataset.make_one_shot_iterator().get_next(),
            lambda: test_dataset.make_one_shot_iterator().get_next()
        )

        self.pixels = tf.cast(self.pixels, tf.float32) / 255.
        self.background = tf.cast(self.background, tf.float32) / 255.


def project_vertices(split_vertices_world, view_matrix, projection_matrix):

    # Transform vertices from world to camera then clip space
    homogeneous_vertices_world = tf.concat([
        split_vertices_world,
        tf.ones_like(split_vertices_world[..., :1])
    ], axis=-1)  # indexed by vertex-index, x/y/z/w
    homogeneous_vertices_camera = tf.matmul(homogeneous_vertices_world, view_matrix)  # indexed by vertex-index, x/y/z/w
    projected_vertices_camera = tf.matmul(homogeneous_vertices_camera, projection_matrix)
    return projected_vertices_camera


def render_scene_bcs(split_vertices_world, split_faces, split_vertex_colours_unlit, background, projection_matrix, view_matrix, clip_to_world_matrix, view_to_world_matrix):

    projected_vertices_camera = project_vertices(split_vertices_world, view_matrix, projection_matrix)

    # Calculate the lighting
    vertex_normals_world = dirt.lighting.vertex_normals_pre_split(split_vertices_world, split_faces)
    diffuse_component = dirt.lighting.diffuse_directional(
        vertex_normals_world,
        split_vertex_colours_unlit,
        light_direction=sun_direction,
        light_color=[1., 1., 1.],
        double_sided=True
    )
    vertex_colours = tf.clip_by_value(diffuse_component * 0.7 + split_vertex_colours_unlit * 0.3, 0., 1.)  # indexed by vertex-index, r/g/b

    # Calculate the basis of the plane we're going to render the view 'from' the sun in. We arbitrarily choose
    # to make e_y point in a similar direction to the world 'up' vector
    # Note that the origin of this plane is unspecified, and doesn't matter as we'll do an orthographic projection
    up_world = np.float32([0., 1., 0.])
    sun_plane_ey_world = up_world - np.dot(up_world, sun_direction) * sun_direction
    sun_plane_ey_world /= np.linalg.norm(sun_plane_ey_world)
    sun_plane_ex_world = np.cross(sun_plane_ey_world, sun_direction)

    # Then, project the geometry and find its bounds wrt this plane
    world_to_sun_matrix = tf.constant(np.stack([sun_plane_ex_world, sun_plane_ey_world, sun_direction], axis=1), dtype=tf.float32)
    projected_vertices_sun = tf.matmul(split_vertices_world, world_to_sun_matrix)
    min_in_sun_plane = tf.reduce_min(projected_vertices_sun, axis=0)
    max_in_sun_plane = tf.reduce_max(projected_vertices_sun, axis=0)
    shadow_map_margin = (max_in_sun_plane - min_in_sun_plane) * (2. / max(shadow_map_width, shadow_map_height))
    min_in_sun_plane -= shadow_map_margin
    max_in_sun_plane += shadow_map_margin

    # Normalise the bounds in the sun plane to [-1, 1], and add unit w-coordinate for orthographic projection
    homogeneous_vertices_sun = tf.concat([
        -1. + 2. * (projected_vertices_sun - min_in_sun_plane) / (max_in_sun_plane - min_in_sun_plane),
        tf.ones_like(projected_vertices_sun[:, :1])
    ], axis=1)

    # Rasterise the shadow map
    sun_pixels = dirt.rasterise(
        vertices=homogeneous_vertices_sun,
        faces=split_faces,
        vertex_colors=tf.ones_like(homogeneous_vertices_sun[:, :1]),
        background=tf.zeros([shadow_map_height, shadow_map_height, 1]),
        width=shadow_map_width, height=shadow_map_height, channels=1,
        name='sun_pixels'
    )[:, :, 0]  # indexed by y, x

    # Build the set of image-space pixel-coordinates, map to ndc, and unproject onto ground-plane
    assert crop_height % shadow_stride == 0 and crop_width % shadow_stride == 0
    pixel_locations_image = tf.cast(tf.stack([
        tf.tile(tf.range(crop_width, delta=shadow_stride)[np.newaxis, :], [crop_height // shadow_stride, 1]),
        tf.tile(tf.range(crop_height, delta=shadow_stride)[:, np.newaxis], [1, crop_width // shadow_stride])
    ], axis=2), tf.float32)  # indexed by y, x, x/y
    pixel_ground_intersections_world = camera_calibration.unproject_onto_ground(pixel_locations_image, clip_to_world_matrix, [crop_width, crop_height])

    # Map points from the ground-plane into the sun-plane, and thence pixel-space of the shadow map
    pixel_ground_intersections_sun = tf.tensordot(pixel_ground_intersections_world, world_to_sun_matrix, axes=1)
    pixel_ground_intersections_sun = ((pixel_ground_intersections_sun - min_in_sun_plane) / (max_in_sun_plane - min_in_sun_plane))[:, :, :2]  # indexed by y, x, x/y
    pixel_ground_intersections_sun = pixel_ground_intersections_sun * [1., -1.] + [0., 1.]
    pixel_ground_intersections_sun = tf.cast(pixel_ground_intersections_sun * [shadow_map_width, shadow_map_height] + 0.5, tf.int32)
    pixel_ground_intersections_sun = tf.clip_by_value(pixel_ground_intersections_sun, 0, [shadow_map_width - 1, shadow_map_height - 1])

    # Gather the relevant pixels from the shadow map; smooth; apply to background
    shadow_pixels = tf.gather_nd(sun_pixels, pixel_ground_intersections_sun[:, :, ::-1])  # this is one in shade, zero in sunlight
    shadow_pixels = tf.image.resize_bilinear(shadow_pixels[np.newaxis, :, :, np.newaxis], [crop_height, crop_width])  # indexed by singleton, y, x, singleton
    kernel_1d = cv2.getGaussianKernel(ksize=17, sigma=7.)
    smoothing_kernel = tf.constant((kernel_1d * kernel_1d.T)[:, :, np.newaxis, np.newaxis], dtype=tf.float32)
    shadow_pixels = tf.nn.conv2d(shadow_pixels, smoothing_kernel, [1, 1, 1, 1], 'SAME')[0, :, :, 0]
    shadowed_background = background * tf.pow(shadow_scaling, shadow_pixels[:, :, np.newaxis])

    # Render the fg objects over the shadowed background
    pixels = dirt.rasterise(
        vertices=projected_vertices_camera,
        faces=split_faces,
        vertex_colors=vertex_colours,
        background=shadowed_background,
        width=crop_width, height=crop_height, channels=3,
        name='pixels'
    )  # indexed by y, x, r/g/b

    silhouette = dirt.rasterise(
        vertices=projected_vertices_camera,
        faces=split_faces,
        vertex_colors=tf.ones_like(vertex_colours[..., :1]),
        background=tf.zeros_like(shadowed_background[..., :1]),
        width=crop_width, height=crop_height, channels=1,
        name='silhouette'
    )[..., 0]  # indexed by y, x

    return pixels, silhouette


def render_scene_shapenet(split_vertices_world, split_faces, split_vertex_colours_unlit, background, projection_matrix, view_matrix, clip_to_world_matrix, view_to_world_matrix):

    projected_vertices_camera = project_vertices(split_vertices_world, view_matrix, projection_matrix)

    # Calculate the lighting
    vertex_normals_world = dirt.lighting.vertex_normals_pre_split(split_vertices_world, split_faces)
    diffuse_component = dirt.lighting.diffuse_directional(
        vertex_normals_world,
        split_vertex_colours_unlit,
        light_direction=sun_direction,
        light_color=[1., 1., 1.],
        double_sided=True
    )
    vertex_colours = tf.clip_by_value(diffuse_component * 0.7 + split_vertex_colours_unlit * 0.3, 0., 1.)  # indexed by vertex-index, r/g/b

    pixels = dirt.rasterise(
        vertices=projected_vertices_camera,
        faces=split_faces,
        vertex_colors=vertex_colours,
        background=background,
        name='pixels'
    )  # indexed by y, x, r/g/b

    silhouette = dirt.rasterise(
        vertices=projected_vertices_camera,
        faces=split_faces,
        vertex_colors=tf.ones_like(vertex_colours[..., :1]),
        background=tf.zeros_like(background[..., :1]),
        name='silhouette'
    )[..., 0]  # indexed by y, x

    return pixels, silhouette


def render_turntable(split_vertices_object, split_faces, split_vertex_colours_unlit):

    # ** note that the rotation, view, and projection matrices could be calculated outside here, to avoid duplication when mapped over iib

    # Generate one set of world-space vertices per orientation
    orientation_step = 2. * np.pi / turntable_orientations
    rotation_matrices = dirt.matrices.rodrigues((tf.range(turntable_orientations, dtype=tf.float32)[:, np.newaxis] + 0.5) * orientation_step * [0., 1., 0.])  # indexed by orientation-index, x/y/z/w (in), x/y/z/w (out)
    homogeneous_vertices_object = tf.concat([split_vertices_object, tf.ones_like(split_vertices_object[:, :1])], axis=1)  # indexed by vertex-index, x/y/z/w
    homogeneous_vertices_world = tf.matmul(tf.tile(homogeneous_vertices_object[np.newaxis, :, :], [turntable_orientations, 1, 1]), rotation_matrices)  # indexed by orientation-index, vertex-index, x/y/z/w

    # Transform vertices from world to camera then clip space
    if dataset == 'cub':
        split_vertices_object *= 2.
    view_matrix = tf.matmul(dirt.matrices.rodrigues([-turntable_camera_elevation, 0., 0.]), dirt.matrices.translation([0., 0., -turntable_camera_distance]))  # indexed by x/y/z/w (in), x/y/z/w (out)
    projection_matrix = dirt.matrices.perspective_projection(near=1., far=50., right=0.3, aspect=float(crop_height) / crop_width)  # indexed by x/y/z/w (in), x/y/z/w (out)
    projected_vertices_camera = tf.tensordot(homogeneous_vertices_world, tf.matmul(view_matrix, projection_matrix), axes=1)  # indexed by orientation-index, vertex-index, x/y/z/w
    projected_vertices_camera.set_shape(homogeneous_vertices_world.get_shape())

    # Calculate the lighting
    vertex_normals_world = dirt.lighting.vertex_normals_pre_split(homogeneous_vertices_world, split_faces)
    replicated_vertex_colours_unlit = tf.tile(split_vertex_colours_unlit[np.newaxis, :], [turntable_orientations, 1, 1])
    diffuse_component = dirt.lighting.diffuse_directional(
        vertex_normals_world,
        replicated_vertex_colours_unlit,
        light_direction=tf.tile(sun_direction[np.newaxis, :], [turntable_orientations, 1]),
        light_color=tf.ones([turntable_orientations, 3]),
        double_sided=True
    )
    vertex_colours = diffuse_component * 0.7 + replicated_vertex_colours_unlit * 0.3  # indexed by orientation-index, vertex-index, r/g/b

    natural_pixels = dirt.rasterise_batch(
        vertices=projected_vertices_camera,
        faces=tf.tile(split_faces[np.newaxis, :], [turntable_orientations, 1, 1]),
        vertex_colors=vertex_colours,
        background=tf.ones([turntable_orientations, crop_height, crop_width, 3]) * [0.7, 0.3, 0.7],
        width=crop_width, height=crop_height, channels=3
    )  # indexed by orientation-index, y, x, r/g/b

    normal_pixels = dirt.rasterise_batch(
        vertices=projected_vertices_camera,
        faces=tf.tile(split_faces[np.newaxis, :], [turntable_orientations, 1, 1]),
        vertex_colors=tf.tile(tf.abs(dirt.lighting.vertex_normals_pre_split(homogeneous_vertices_object, split_faces))[np.newaxis, :, :], [turntable_orientations, 1, 1]),
        background=tf.zeros([turntable_orientations, crop_height, crop_width, 3]),
        width=crop_width, height=crop_height, channels=3
    )  # indexed by orientation-index, y, x, r/g/b

    pixels = tf.concat([
        tf.reshape(tf.transpose(natural_pixels, [1, 0, 2, 3]), [crop_height, turntable_orientations * crop_width, 3]),  # indexed by y, orientation-index * x, r/g/b
        tf.reshape(tf.transpose(normal_pixels, [1, 0, 2, 3]), [crop_height, turntable_orientations * crop_width, 3]),  # ditto
    ], axis=0)  # indexed by natural/normals * y, orientation-index * x, r/g/b

    return pixels


def get_crease_loss(vertices, faces):

    # vertices is indexed by iib, vertex-index, x/y/z

    creases = tf.constant(get_creases(faces), dtype=tf.int32)
    first_endpoints, second_endpoints, lefts, rights = tf.map_fn(
        lambda vertices_for_iib: (
            tf.gather(vertices_for_iib, creases[:, 0]),
            tf.gather(vertices_for_iib, creases[:, 1]),
            tf.gather(vertices_for_iib, creases[:, 2]),
            tf.gather(vertices_for_iib, creases[:, 3]),
        ),
        vertices, (tf.float32, tf.float32, tf.float32, tf.float32)
    )  # each indexed by iib, crease-index, x/y/z

    line_displacements = second_endpoints - first_endpoints
    line_directions = line_displacements / tf.norm(line_displacements, axis=-1, keep_dims=True)
    lefts_projected = first_endpoints + tf.reduce_sum((lefts - first_endpoints) * line_directions, axis=-1, keep_dims=True) * line_directions
    rights_projected = first_endpoints + tf.reduce_sum((rights - first_endpoints) * line_directions, axis=-1, keep_dims=True) * line_directions

    left_directions = lefts - lefts_projected
    left_directions /= tf.norm(left_directions, axis=-1, keep_dims=True)
    right_directions = rights - rights_projected
    right_directions /= tf.norm(right_directions, axis=-1, keep_dims=True)
    cosines = tf.reduce_sum(left_directions * right_directions, axis=-1)
    loss = tf.reduce_mean(tf.sqrt(cosines + 1. + 1.e-6))

    return loss


def get_l2_laplacian_loss(vertices, faces):

    adjacency_matrix = get_adjacency_matrix(faces).astype(np.float32)
    laplacian = np.eye(adjacency_matrix.shape[0]) - adjacency_matrix / np.sum(adjacency_matrix, axis=1, keepdims=True)
    laplacian = tf.constant(laplacian, dtype=tf.float32)
    delta_coordinates = tf.matmul(tf.tile(laplacian[np.newaxis, :, :], [int(vertices.get_shape()[0]), 1, 1]), vertices)  # indexed by iib, vertex-index, x/y/z
    delta_norms = tf.norm(delta_coordinates, axis=2, ord=2)
    return tf.reduce_mean(tf.reduce_max(delta_norms, axis=1)) * 1. + tf.reduce_mean(delta_norms) * 1.


def get_adjacent_vertex_displacements(vertices, faces):

    adjacency_matrix = get_adjacency_matrix(faces)  # indexed by first-vertex, second-vertex
    adjacency_list = np.asarray([(first, second) for first, second in np.transpose(np.where(adjacency_matrix)) if first < second])  # indexed by undirected-edge-index, first-/second-vertex
    adjacent_vertex_pairs = tf.map_fn(
        lambda vertices_for_iib: tf.gather(vertices_for_iib, adjacency_list),
        vertices
    )  # indexed by iib, edge-index, first-/second-vertex, x/y/z
    adjacent_vertex_displacements = adjacent_vertex_pairs[:, :, 1, :] - adjacent_vertex_pairs[:, :, 0, :]
    return adjacent_vertex_displacements


def get_l2_grad_loss(vertices, faces):

    adjacent_vertex_displacements = get_adjacent_vertex_displacements(vertices, faces)
    return tf.reduce_mean(tf.norm(adjacent_vertex_displacements, axis=-1))


def get_tv_l1_loss(vertices, faces):

    # vertices is indexed by iib, vertex-index, x/y/z
    adjacent_vertex_displacements = get_adjacent_vertex_displacements(vertices, faces)
    return tf.reduce_mean(tf.abs(adjacent_vertex_displacements))


def get_equilaterality_loss(vertices, faces):

    triangles = tf.transpose(tf.gather(vertices, faces, axis=1), [1, 0, 2, 3])  # :: iib, face, vertex-in-face, x/y/z

    edge_displacements = tf.stack([
        triangles[:, :, 1] - triangles[:, :, 0],
        triangles[:, :, 2] - triangles[:, :, 1],
        triangles[:, :, 0] - triangles[:, :, 2],
    ], axis=2)  # :: iib, face, edge-in-face, x/y/z

    edge_lengths = tf.linalg.norm(edge_displacements, axis=-1)
    max_lengths = tf.reduce_max(edge_lengths, axis=-1)  # :: iib, face
    min_lengths = tf.reduce_min(edge_lengths, axis=-1)

    return tf.reduce_mean(max_lengths / (min_lengths + 1.e-4))


class Generative(object):

    vertex_anchors_and_faces = None
    shape_model_scope = None

    get_vertices_memoised_results = {}

    @staticmethod
    def get_vertices(shape_embedding, scale, rotation):

        memo_key = shape_embedding.name, scale.name, rotation.name
        if memo_key in Generative.get_vertices_memoised_results:
            print('get_vertices returning memoised result for ' + str(memo_key))
            return Generative.get_vertices_memoised_results[memo_key]

        if Generative.vertex_anchors_and_faces is None:

            if base_mesh == 'cube':
                vertex_anchors, faces = Generative.build_simple_subdivided_cube(subdivision_count)  # vertex_anchors is indexed by vertex-index, x/y/z
            elif base_mesh == 'sphere':
                vertex_anchors, faces = meshzoo.iso_sphere(subdivision_count)
                faces = point_normals_outward(vertex_anchors, faces)
                vertex_anchors *= sphere_scale
            else:
                assert False
            vertex_anchors = np.float32(np.asarray(vertex_anchors) * anchor_scaling / 2.)  # divide by two as the 'raw' cube is [-1, 1]^3
            Generative.vertex_anchors_and_faces = vertex_anchors, faces
            print('{} vertices, {} faces'.format(len(vertex_anchors), len(faces)))

        else:

            vertex_anchors, faces = Generative.vertex_anchors_and_faces

        vertex_count = len(vertex_anchors)

        def build_shape_model():

            with slim.arg_scope([slim.fully_connected], variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES, 'gen_shape_decoder']):
                if shape_model == 'FA':

                    mean_shape = slim.model_variable('mean_shape', initializer=vertex_anchors, collections=[tf.GraphKeys.TRAINABLE_VARIABLES, 'gen_shape_decoder'])  # indexed by vertex-in-object, x/y/z
                    delta_shape = slim.fully_connected(shape_embedding, vertex_count * 3, activation_fn=None, weights_initializer=tf.zeros_initializer(), biases_initializer=None)  # indexed by iib, object-index, vertex-in-object * x/y/z
                    delta_shape = tf.reshape(delta_shape, [-1, vertex_count, 3])  # indexed by iib, vertex-index, x/y/z
                    return mean_shape + delta_shape  # indexed by iib, vertex-index, x/y/z

                elif shape_model == 'VAE':

                    shape_decoded = slim.fully_connected(shape_embedding, 32, activation_fn=tf.nn.elu)
                    vertex_perturbations = tf.reshape(
                        slim.fully_connected(shape_decoded, vertex_count * 3, weights_initializer=tf.zeros_initializer(), activation_fn=None),
                        [-1, vertex_count, 3]
                    )  # indexed by iib, vertex-index, x/y/z
                    return tf.constant(vertex_anchors, dtype=tf.float32) + vertex_perturbations

                elif shape_model == 'VAE-seq-att':

                    shape_decoded = slim.fully_connected(shape_embedding, 128, activation_fn=tf.nn.elu)
                    shape_decoded += slim.fully_connected(shape_decoded, 128, activation_fn=tf.nn.elu)
                    shape_decoded += slim.fully_connected(shape_decoded, 128, activation_fn=tf.nn.elu)

                    vertices = tf.tile(tf.constant(vertex_anchors, dtype=tf.float32)[None, :, :], [images_per_batch, 1, 1])
                    for offset_index in range(sequential_attentive_steps):

                        vertex_attention_map = slim.fully_connected(shape_decoded, vertex_count, activation_fn=tf.nn.sigmoid)  # indexed by iib, vertex-index
                        offset = slim.fully_connected(shape_decoded, 3, weights_initializer=tf.zeros_initializer(), activation_fn=None)  # indexed by iib, x/y/z
                        weighted_offsets = vertex_attention_map[:, :, None] * offset[:, None, :]  # indexed by iib, vertex-index, x/y/z

                        vertices += weighted_offsets

                    return vertices

                elif shape_model == 'VAE-seq-att-pushing':

                    from mesh_intersections.mesh_pushing import apply_offsets_with_pushing  # delayed import so we don't require gurobi if using other parameterisations

                    shape_decoded = slim.fully_connected(shape_embedding, 128, activation_fn=tf.nn.elu)
                    shape_decoded += slim.fully_connected(shape_decoded, 128, activation_fn=tf.nn.elu)
                    shape_decoded += slim.fully_connected(shape_decoded, 128, activation_fn=tf.nn.elu)

                    def build_pushed_vertices(relevant_vertices, relevant_faces):  # 'relevant' may be half of them if in mirrored mode

                        relevant_vertex_count = relevant_vertices.get_shape()[1].value
                        for offset_index in range(sequential_attentive_steps):

                            initial_offset_magnitudes = slim.fully_connected(shape_decoded, relevant_vertex_count, activation_fn=tf.nn.softplus) * 0.1  # indexed by iib, vertex-index
                            offset_direction_unnormalised = slim.fully_connected(shape_decoded, 3, activation_fn=None)  # indexed by iib, x/y/z

                            # ** how to make it a unit vector but smoothly parameterised? quaterions (c.f. QuaterNet)?
                            offset_direction = tf.linalg.l2_normalize(offset_direction_unnormalised, axis=-1)

                            relevant_vertices = apply_offsets_with_pushing(relevant_vertices, relevant_faces, offset_direction, initial_offset_magnitudes)

                        return relevant_vertices

                    if flip_augmentation:  # assume we also want the model to be z-symmetric in this case:

                        assert base_mesh == 'cube' and subdivision_count % 2 == 1  # ...as we assume there is one matching vertex in the right half for each in the left!

                        left_vertex_indicators = vertex_anchors[:, 2] <= 0.
                        right_vertex_indicators = np.logical_not(left_vertex_indicators)
                        index_in_lefts_to_index_in_all = np.where(left_vertex_indicators)[0]
                        index_in_rights_to_index_in_all = np.where(right_vertex_indicators)[0]
                        left_vertices = vertex_anchors[left_vertex_indicators]
                        right_vertices = vertex_anchors[right_vertex_indicators]
                        left_faces = tf.constant(list(filter(
                            lambda face: left_vertex_indicators[face[0]] and left_vertex_indicators[face[1]] and left_vertex_indicators[face[2]],
                            faces
                        )), dtype=tf.int32)

                        left_vertices_pushed = build_pushed_vertices(
                            tf.tile(tf.constant(left_vertices, dtype=tf.float32)[None, :, :], [images_per_batch, 1, 1]),
                            left_faces
                        )  # :: iib, left-vertex, x/y/z

                        max_positive_left_z_pushed_or_zero = tf.nn.relu(tf.reduce_max(left_vertices_pushed[:, :, 2], axis=1) + 0.05)  # :: iib
                        left_vertices_pushed = tf.concat([
                            left_vertices_pushed[:, :, :2],
                            left_vertices_pushed[:, :, 2:] - max_positive_left_z_pushed_or_zero[:, None, None]
                        ], axis=2)

                        left_to_mirrored_right_distances = np.linalg.norm(right_vertices[None, :, :] * [1, 1, -1] - left_vertices[:, None, :], axis=-1)  # :: left-vertex, right-vertex
                        index_in_lefts_to_mirroring_index_in_rights = np.argmin(left_to_mirrored_right_distances, axis=1)  # :: left-vertex
                        index_in_lefts_to_mirroring_index_in_all = index_in_rights_to_index_in_all[index_in_lefts_to_mirroring_index_in_rights]

                        vertices = tf.transpose(
                            tf.scatter_nd(index_in_lefts_to_index_in_all[:, None], tf.transpose(left_vertices_pushed, [1, 0, 2]), [vertex_count, images_per_batch, 3]) +
                                tf.scatter_nd(index_in_lefts_to_mirroring_index_in_all[:, None], tf.transpose(left_vertices_pushed * [1, 1, -1], [1, 0, 2]), [vertex_count, images_per_batch, 3]),
                            [1, 0, 2]
                        )
                        return vertices

                    else:

                        return build_pushed_vertices(
                            tf.tile(tf.constant(vertex_anchors, dtype=tf.float32)[None, :, :], [images_per_batch, 1, 1]),
                            tf.constant(faces, tf.int32)
                        )

                else:
                    assert False

        if Generative.shape_model_scope is None:
            with tf.variable_scope('shape_model', reuse=tf.AUTO_REUSE) as scope:
                vertices_object_unscaled = build_shape_model()
                Generative.shape_model_scope = scope
        else:
            with tf.variable_scope(Generative.shape_model_scope):
                vertices_object_unscaled = build_shape_model()

        if flip_augmentation:
            flip_shape = Bernoulli(tf.ones([images_per_batch]) * 0.5)
            flip_indicator = tf.cast(flip_shape, tf.float32) * 2. - 1.
            vertices_object = vertices_object_unscaled * (flip_indicator[:, np.newaxis, np.newaxis] * [0., 0., 1.] + [1., 1., 0.])
        else:
            vertices_object = vertices_object_unscaled
        vertices_object *= scale[:, np.newaxis, :]

        vertical_rotation_matrix = dirt.matrices.rodrigues(rotation[:, np.newaxis] * [0., 1., 0.])[:, :3, :3]  # indexed by iib, x/y/z (in), x/y/z (out)
        vertices_world = tf.matmul(vertices_object, vertical_rotation_matrix)  # indexed by iib, vertex-index, x/y/z

        result = vertices_object_unscaled, vertices_object, vertices_world, faces
        Generative.get_vertices_memoised_results[memo_key] = result
        return result

    @staticmethod
    def get_view_matrix(conditioning):

        if dataset == 'bcs':

            camera_height = tf.ones([images_per_batch]) * default_camera_height
            camera_locations_homogeneous = camera_height[:, np.newaxis] * [0., 1., 0., 0.] + [0., 0., 0., 1.]  # indexed by iib, x/y/z/w
            view_translation = -tf.matmul(camera_locations_homogeneous[:, np.newaxis, :], conditioning.view_rotation_matrix)[:, 0, :3]
            view_translation_matrix = dirt.matrices.translation(view_translation)
            return tf.matmul(conditioning.view_rotation_matrix, view_translation_matrix)

        elif dataset == 'shapenet' or dataset == 'cub':

            return conditioning.view_rotation_matrix  # in fact the conditioning stores a full view matrix in this case

        else:

            assert False

    def __init__(self, is_training, use_gt_background, conditioning, rv, mode):

        # background is indexed by y, x, r/g/b, and assumed to be of the correct size!

        view_matrix = self.get_view_matrix(conditioning)
        clip_to_world_matrix = tf.matrix_inverse(tf.matmul(view_matrix, conditioning.projection_matrix))
        view_to_world_matrix = tf.matrix_inverse(view_matrix)

        # ** scale prior should have a single learnt parameter vector, which hopefully learns to make one axis long and the other narrow
        self.scale = rv('scale', lambda sample_count: Normal(tf.ones([sample_count * images_per_batch, 3]) * 1., 0.05, name='scale'))
        if joint_embedding_dimensionality is None:
            self.shape_embedding = rv(
                'shape_embedding',
                lambda sample_count: Normal(tf.zeros([sample_count * images_per_batch, shape_embedding_dimensionality]), 1., name='shape_embedding')
            )  # indexed by iib, latent-channel
            self.colour_embedding = rv(
                'colour_embedding',
                lambda sample_count: Normal(tf.zeros([sample_count * images_per_batch, colour_embedding_dimensionality]), 1., name='colour_embedding')
            )  # indexed by iib, latent-channel
            self.colour_embedding += slim.fully_connected(tf.concat([self.colour_embedding, self.shape_embedding], axis=-1), colour_embedding_dimensionality, activation_fn=tf.nn.elu, variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES, 'gen_colour_decoder'])
        else:
            self.joint_embedding = rv(
                'joint_embedding',
                lambda sample_count: Normal(tf.zeros([sample_count * images_per_batch, joint_embedding_dimensionality]), 1., name='joint_embedding')
            )  # indexed by iib, latent-channel
            self.shape_embedding = slim.fully_connected(self.joint_embedding, shape_embedding_dimensionality, activation_fn=tf.nn.elu, variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES, 'gen_shape_decoder'])
            self.colour_embedding = slim.fully_connected(self.joint_embedding, colour_embedding_dimensionality, activation_fn=tf.nn.elu)

        vertices_object_unscaled, self.vertices_object, vertices_world, faces = self.get_vertices(self.shape_embedding, self.scale, conditioning.rotation)

        if mode in {GenerativeMode.CONDITIONED}:
            if l2_laplacian_strength > 0.:
                tf.losses.add_loss(get_l2_laplacian_loss(vertices_object_unscaled, faces) * l2_laplacian_strength)
            if tv_l1_strength > 0:
                tf.losses.add_loss(get_tv_l1_loss(vertices_object_unscaled, faces) * tv_l1_strength)
            if l2_grad_strength > 0.:
                tf.losses.add_loss(get_l2_grad_loss(vertices_object_unscaled, faces) * l2_grad_strength)
            if crease_strength > 0.:
                tf.losses.add_loss(get_crease_loss(vertices_object_unscaled, faces) * crease_strength)
            if equilat_strength > 0.:
                tf.losses.add_loss(get_equilaterality_loss(vertices_object_unscaled, faces) * equilat_strength)

        if dataset == 'bcs':
            # For BCS, place the centre of the object at the world-space location under the centre of the crop
            unprojected_centre = tf.map_fn(
                lambda clip_to_world_matrix_for_iib:
                    # The * 0.7 here is because the centre of a crop is the centre of the silhouette, which is always higher than the centre of the car's base
                    camera_calibration.unproject_onto_ground(tf.constant([crop_width / 2., crop_height * 0.7], tf.float32), clip_to_world_matrix_for_iib, [crop_width, crop_height]),
                clip_to_world_matrix,
                dtype=tf.float32, back_prop=False
            )
            y_location = -tf.reduce_min(self.vertices_object[:, :, 1], axis=1)  # ensure the base of each object rests on the ground; indexed by iib
            self.xz_offset = rv('xz_offset', lambda sample_count: Normal(tf.zeros([sample_count * images_per_batch, 2]), 0.25, name='xz_offset'))
            locations = tf.stack([self.xz_offset[:, 0], y_location, self.xz_offset[:, 1]], axis=1) + unprojected_centre  # indexed by iib, x/y/z
            vertices_world += locations[:, np.newaxis, :]

        split_vertices_world, self.split_faces = dirt.lighting.split_vertices_by_face(vertices_world, faces)
        self.split_vertices_object, _ = dirt.lighting.split_vertices_by_face(self.vertices_object, faces)

        with slim.arg_scope([slim.fully_connected], variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES, 'gen_colour_decoder']):
            colours_decoded = slim.fully_connected(self.colour_embedding, colour_decoder_hidden_dimensionality, activation_fn=tf.nn.elu)
            self.face_colours = tf.clip_by_value(tf.reshape(
                slim.fully_connected(colours_decoded, len(faces) * 3, activation_fn=None),
                [-1, len(faces), 3]
            ) * 0.1 + 0.5, 0., 1.)  # indexed by iib, face-index, r/g/b

        if background_embedding_dimensionality is None:
            self.background = conditioning.background
        else:
            self.background_embedding = rv(
                'background_embedding',
                lambda sample_count: Normal(tf.zeros([sample_count * images_per_batch, background_embedding_dimensionality]), 1., name='background_embedding')
            )
            self.background = tf.cond(
                use_gt_background,
                lambda: conditioning.background,  # gives pretty images
                lambda: self.build_background_net(self.background_embedding)  # used for training
            )

        self.vertex_colours = tf.reshape(tf.tile(self.face_colours[:, :, np.newaxis, :], [1, 1, 3, 1]), [-1, len(faces) * 3, 3])  # indexed by iib, face-index * vertex-in-face, r/g/b

        def render_iib(parameters):
            (
                vertices_world_for_iib, vertex_colours_for_iib,
                background_for_iib,
                projection_matrix_for_iib, view_matrix_for_iib, clip_to_world_matrix_for_iib, view_to_world_matrix_for_iib
            ) = parameters
            return (render_scene_bcs if dataset == 'bcs' else render_scene_shapenet)(
                vertices_world_for_iib, self.split_faces, vertex_colours_for_iib,
                background_for_iib,
                projection_matrix_for_iib, view_matrix_for_iib, clip_to_world_matrix_for_iib, view_to_world_matrix_for_iib
            )
        self.pixels, self.silhouette = tf.map_fn(
            render_iib, [
                split_vertices_world, self.vertex_colours, self.background,
                conditioning.projection_matrix, view_matrix, clip_to_world_matrix, view_to_world_matrix
            ], dtype=(tf.float32, tf.float32)
        )
        self.colour_pyramid = rv('colour_pyramid', lambda: SsimPyramid(self.pixels, name='colour_pyramid'))
        if background_embedding_dimensionality is None:
            self.silhouette_pyramid = rv('silhouette_pyramid', lambda: SilhouettePyramid(self.silhouette, levels=6, operation='iou_pyr', name='silhouette_pyramid'))

        self.turntable_pixels = tf.map_fn(
            lambda split_vertices_object_and_vertex_colours_for_iib:
                render_turntable(split_vertices_object_and_vertex_colours_for_iib[0], self.split_faces, split_vertices_object_and_vertex_colours_for_iib[1]),
            [self.split_vertices_object[:recon_image_count], self.vertex_colours[:recon_image_count]],
            dtype=tf.float32
        )

    @staticmethod
    def build_simple_subdivided_cube(K):

        # Build the faces, assuming 'all' vertices, i.e. including interior grid points, are present, and numbered
        # by increasing z, then y, then x (fastest). Then, trim the set of vertices to the interior, and adjust faces
        # accordingly. See notes p37
        assert K > 0
        quad = lambda a, b, c, d: [[a, b, d], [b, c, d]]

        z_face_row = lambda first: sum([
            quad(quad_first, quad_first + 1, quad_first + K + 2, quad_first + K + 1)
            for quad_first in range(first, first + K)
        ], [])
        z_face = lambda first: sum([z_face_row(first + y * (K + 1)) for y in range(K)], [])

        y_face_row = lambda first: sum([
            quad(
                quad_first,
                quad_first + (K + 1) * (K + 1),
                quad_first + (K + 1) * (K + 1) + 1,
                quad_first + 1
            )
            for quad_first in range(first, first + K)
        ], [])
        y_face = lambda first: sum([y_face_row(first + z * (K + 1) * (K + 1)) for z in range(K)], [])

        x_face_row = lambda first: sum([
            quad(
                quad_first,
                quad_first + (K + 1) * (K + 1),
                quad_first + (K + 2) * (K + 1),
                quad_first + K + 1
            )
            for quad_first in range(first, first + (K + 1) * (K + 1) * K, (K + 1) * (K + 1))
        ], [])
        x_face = lambda first: sum([x_face_row(first + y * (K + 1)) for y in range(K)], [])

        def flip(triangles):
            return list(map(reversed, triangles))
        faces = flip(z_face(0)) + z_face(K * (K + 1) * (K + 1)) + flip(y_face(0)) + y_face(K * (K + 1)) + x_face(0) + flip(x_face(K))

        spacing = 2. / K
        vertices = []
        for z in range(K + 1):
            for y in range(K + 1):
                for x in range(K + 1):
                    if x != 0 and y != 0 and z != 0 and x != K and y != K and z != K:  # interior
                        faces = [[index if index < len(vertices) else index - 1 for index in face] for face in faces]
                    else:  # exterior
                        vertices.append([x * spacing - 1., y * spacing - 1., z * spacing - 1.])

        return vertices, faces

    @staticmethod
    def build_background_net(embedding):

        def upsample(x, factor=2):
            return tf.image.resize_bilinear(x, [x.get_shape()[1].value * factor, x.get_shape()[2].value * factor], align_corners=True)

        net = upsample(upsample(tf.reshape(embedding, [images_per_batch, 1, 1, background_embedding_dimensionality])))
        net = upsample(slim.conv2d(net, 64, kernel_size=3, activation_fn=tf.nn.elu))
        net = upsample(slim.conv2d(net, 32, kernel_size=3, activation_fn=tf.nn.elu))
        net = upsample(slim.conv2d(net, 16, kernel_size=3, activation_fn=tf.nn.elu))
        net = slim.conv2d(net, 3, kernel_size=3, activation_fn=None)
        net = upsample(net, crop_width // 32)

        return tf.sigmoid(net * 0.5)


class Variational(object):

    def __init__(self, is_training, ground_truth, rv, observation):

        pixels = observation('colour_pyramid')

        # ** following should presumably allow for sample-/discrete-expansion, rather than assuming images_per_batch as first dimension
        features = Variational.build_cnn(pixels, ground_truth.rotation, is_training)  # indexed by iib, channel

        if dataset == 'bcs':
            xz_offset_mean = tf.reshape(tf.tanh(slim.fully_connected(features, 2, activation_fn=None) * 0.1) * 4., [images_per_batch, 2])  # indexed by iib, x/z
            xz_offset_sigma = tf.reshape(slim.fully_connected(features, 2, activation_fn=tf.nn.softplus), [images_per_batch, 2]) * 0. + 0.01  # ditto
            self.xz_offset = rv('xz_offset', lambda: Normal(xz_offset_mean, xz_offset_sigma, name='q_xz_offset'))

        scale_mean = tf.reshape(slim.fully_connected(features, 3, activation_fn=None), [images_per_batch, 3])
        scale_mean = tf.pow(2., tf.tanh(scale_mean * 0.1))
        scale_sigma = tf.ones_like(scale_mean) * 0.01
        self.scale = rv('scale', lambda: Normal(scale_mean, scale_sigma, name='q_scale'))

        if background_embedding_dimensionality is not None:
            background_features = slim.fully_connected(features, 64, activation_fn=tf.nn.relu, normalizer_fn=slim.group_norm)
            background_mean = tf.reshape(slim.fully_connected(background_features, background_embedding_dimensionality, activation_fn=None), [images_per_batch, background_embedding_dimensionality])
            background_sigma = tf.reshape(slim.fully_connected(background_features, background_embedding_dimensionality, activation_fn=tf.nn.softplus) + 1.e-6, [images_per_batch, background_embedding_dimensionality])
            self.background_embedding = rv('background_embedding', lambda: Normal(background_mean, background_sigma, name='q_background_embedding'))

        if joint_embedding_dimensionality is None:

            shape_mean = tf.reshape(slim.fully_connected(features, shape_embedding_dimensionality, activation_fn=None), [images_per_batch, shape_embedding_dimensionality])  # indexed by iib, latent-dimension
            shape_sigma = tf.reshape(slim.fully_connected(features, shape_embedding_dimensionality, activation_fn=tf.nn.softplus) + 0.01, [images_per_batch, shape_embedding_dimensionality])  # ditto
            self.shape_embedding = rv('shape_embedding', lambda: Normal(shape_mean, shape_sigma, name='q_shape_embedding'))

            _, _, vertices_world, faces = Generative.get_vertices(self.shape_embedding, self.scale, ground_truth.rotation)
            split_vertices_world, split_faces = dirt.lighting.split_vertices_by_face(vertices_world, faces)
            view_matrix = Generative.get_view_matrix(ground_truth)
            split_vertices_clip = project_vertices(split_vertices_world, view_matrix, ground_truth.projection_matrix)

            face_pseudo_colours = tf.tile(tf.range(len(faces))[None, :, None, None], [images_per_batch, 1, 3, 1])
            vertex_pseudo_colours = tf.reshape(face_pseudo_colours, [images_per_batch, -1, 1])  # indexed by iib, face-index * vertex-in-face, r/g/b
            face_indices = tf.cast(dirt.rasterise_batch(
                vertices=split_vertices_clip,
                faces=tf.tile(split_faces[None], [images_per_batch, 1, 1]),
                vertex_colors=tf.cast(vertex_pseudo_colours, tf.float32),
                background=tf.ones([images_per_batch, crop_height, crop_width, 1]) * len(faces)  # thus, all non-clipping pixels are sent to the (-1)^th face
            ) + 0.1, tf.int32)  # indexed by iib, y, x, singleton

            if background_embedding_dimensionality is None:
                mask_int32 = tf.cast(ground_truth.mask, tf.int32)[:, :, :, None]
                face_indices = face_indices * mask_int32 + len(faces) * (1 - mask_int32)  # thus, all pixels outside the gt mask are sent to the (-1)^th face

            unprojected_colours_and_normalisation = tf.scatter_nd(
                tf.concat([
                    tf.tile(tf.range(images_per_batch)[:, None, None, None], [1, crop_height, crop_width, 1]),
                    face_indices
                ], axis=-1),
                tf.concat([pixels, tf.ones([images_per_batch, crop_height, crop_width, 1])], axis=-1),
                [images_per_batch, len(faces) + 1, 4]
            )  # indexed by iib, face-index, r/g/b/count
            unprojected_colours = tf.concat([
                unprojected_colours_and_normalisation[:, :, :3] / (unprojected_colours_and_normalisation[:, :, 3:] + 1.e-6),
                tf.sign(unprojected_colours_and_normalisation[:, :, 3:])
            ], axis=-1)  # indexed by iib, face-index, r/g/b/mask

            with slim.arg_scope([slim.fully_connected], variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES, 'var_colour_encoder']):

                encoded_unprojection = slim.fully_connected(tf.reshape(unprojected_colours, [images_per_batch, -1]), 96, activation_fn=tf.nn.relu, normalizer_fn=slim.group_norm)
                colour_features = tf.concat([encoded_unprojection, features], axis=-1)

                colour_mean = tf.reshape(slim.fully_connected(colour_features, colour_embedding_dimensionality, activation_fn=None), [images_per_batch, colour_embedding_dimensionality])  # indexed by iib, latent-dimension
                colour_sigma = tf.reshape(slim.fully_connected(colour_features, colour_embedding_dimensionality, activation_fn=tf.nn.softplus) + 0.01, [images_per_batch, colour_embedding_dimensionality])  # ditto

            self.colour_embedding = rv('colour_embedding', lambda: Normal(colour_mean, colour_sigma, name='q_colour_embedding'))

        else:
            embedding_mean = tf.reshape(slim.fully_connected(features, joint_embedding_dimensionality, activation_fn=None), [images_per_batch, joint_embedding_dimensionality])  # indexed by iib, latent-dimension
            embedding_sigma = tf.reshape(slim.fully_connected(features, joint_embedding_dimensionality, activation_fn=tf.nn.softplus) + 0.01, [images_per_batch, joint_embedding_dimensionality])  # ditto
            self.joint_embedding = rv('joint_embedding', lambda: Normal(embedding_mean, embedding_sigma, name='q_joint_embedding'))

    @staticmethod
    def build_cnn(pixels, rotations, is_training):

        # pixels is indexed by iib, y, x, r/g/b
        # rotations is indexed by iib
        # result is indexed by iib, channel

        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            # activation_fn=tf.nn.elu,
            normalizer_fn=slim.group_norm,
        ):
            net = slim.conv2d(pixels, 32, kernel_size=3, stride=2)
            net = slim.conv2d(net, 64, kernel_size=3, stride=1)
            net = slim.max_pool2d(net, kernel_size=2, stride=2)
            centroids_and_rotations = tf.concat([tf.cos(rotations[:, np.newaxis]), tf.sin(rotations[:, np.newaxis])], axis=1)
            net = tf.concat([net, tf.tile(centroids_and_rotations[:, np.newaxis, np.newaxis, :], [1, int(net.get_shape()[1]), int(net.get_shape()[2]), 1])], axis=3)
            net = slim.conv2d(net, 96, kernel_size=3, stride=1)
            net = slim.max_pool2d(net, kernel_size=2, stride=2)
            net = slim.conv2d(net, 128, kernel_size=3, stride=1)
            net = slim.max_pool2d(net, kernel_size=2, stride=2)
            net = slim.conv2d(net, 128, kernel_size=4, stride=1, padding='VALID')

            net = slim.flatten(net)
            net = slim.fully_connected(net, 128)

        return net


def build_param_string():

    if base_mesh == 'sphere':
        param_string = '{}-sphere-{}x'.format(subdivision_count, sphere_scale)
    elif base_mesh == 'cube':
        param_string = '{}-cube'.format(subdivision_count)
    else:
        assert False
    param_string += '_nospec'
    param_string += '_' + shape_model
    if shape_model.startswith('VAE-seq-att'):
        param_string += '-{}-step'.format(sequential_attentive_steps)
        if flip_augmentation:
            param_string += '-sym'

    if joint_embedding_dimensionality is not None:
        param_string += '_{}-elu-jt-e_{}-{}-{}-sce'.format(joint_embedding_dimensionality, shape_embedding_dimensionality, colour_embedding_dimensionality, colour_decoder_hidden_dimensionality)
    else:
        param_string += '_{}-elu-cond-e_{}-{}-sce'.format(shape_embedding_dimensionality, colour_embedding_dimensionality, colour_decoder_hidden_dimensionality)

    param_string += '_2e-1-beta'
    param_string += '_group-norm'

    def short_float(x):
        return np.format_float_scientific(x, trim='-', exp_digits=1).replace('+', '')

    def add_regulariser_string(mnemonic, strength):
        nonlocal param_string
        if strength > 0.:
            param_string += '_{}-{}'.format(short_float(strength), mnemonic)

    add_regulariser_string('L2-lapl', l2_laplacian_strength)
    add_regulariser_string('TV-L1', tv_l1_strength)
    add_regulariser_string('crease', crease_strength)
    add_regulariser_string('L2-grad', l2_grad_strength)
    add_regulariser_string('equilat', equilat_strength)

    param_string += '_no-offset'
    if flip_augmentation:
        param_string += '_flip'
    if background_embedding_dimensionality is None:
        param_string += '_tt_det-sil-iou-pyr'
    else:
        param_string += '_bg-vae-ed-{}-sig'.format(background_embedding_dimensionality)

    param_string += '_{}-seed'.format(random_seed)

    return param_string


def main():

    param_string = build_param_string()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' {}'.format(os.getenv('SLURM_JOB_ID', os.getpid()))
    print('output subfolder: ' + dataset_string + '/' + param_string + '/' + timestamp)
    all_images_path = output_path + '/images/' + dataset_string + '/' + param_string + '/' + timestamp
    latest_images_path = output_path + '/images/' + dataset_string + '/latest'
    latest_image_filename = latest_images_path + '/' + param_string + '.jpg'

    if not os.path.exists(all_images_path):
        os.makedirs(all_images_path)
        os.makedirs(all_images_path + '/meshes')
    if not os.path.exists(latest_images_path):
        os.makedirs(latest_images_path)

    is_training = tf.placeholder(tf.bool, shape=[])  # used to select dataset and switch batch-normalisation mode
    if dataset == 'bcs':
        ground_truth = BcsGroundTruth(is_training)
    elif dataset == 'shapenet':
        ground_truth = ShapenetGroundTruth(is_training)
    elif dataset == 'cub':
        ground_truth = CubGroundTruth(is_training)
    else:
        assert False

    use_gt_background = tf.placeholder(tf.bool, shape=[])  # if set, generated/reconstructed images use the gt (conditioning) background instead of bg-vae; this is used for eval images in bg-vae mode
    loss, grads_and_vars, generative, reconstruction_modes, conditioned_generative = noncopying_integrated_reparam_klqp(
        lambda *args: Generative(is_training, use_gt_background, ground_truth, *args),
        lambda *args: Variational(is_training, ground_truth, *args),
        dict([('colour_pyramid', ground_truth.pixels)] + (
            [('silhouette_pyramid', tf.cast(ground_truth.mask, tf.float32))] if background_embedding_dimensionality is None
            else []
        )),
        {},
        sample_count=1,
        beta=2.e-1,
        grad_clip_magnitude=5.
    )

    global_step = tf.train.create_global_step()
    optimiser = tf.train.AdamOptimizer(1.e-3)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimiser.apply_gradients(grads_and_vars, global_step)

    val_summaries = tf.summary.scalar('validation/loss', loss)
    writer = tf.summary.FileWriter(output_path + '/logs/' + timestamp, graph=tf.get_default_graph(), flush_secs=20)

    assert recon_image_count <= images_per_batch
    overview_images = tf.reshape(tf.concat([
        tf.concat([
            tf.concat([
                ground_truth.pixels[:recon_image_count],
                reconstruction_modes.pixels[:recon_image_count],
            ], axis=2),
            tf.concat([
                tf.cast(ground_truth.mask, tf.float32)[:recon_image_count, :, :, np.newaxis] * ground_truth.pixels[:recon_image_count],
                tf.cast(ground_truth.mask, tf.float32)[:recon_image_count, :, :, np.newaxis] * [0, 1, 0] + reconstruction_modes.silhouette[:recon_image_count, :, :, np.newaxis] * [1, 0, 0]
            ], axis=2)
            if background_embedding_dimensionality is None else
            tf.concat([
                reconstruction_modes.silhouette[:recon_image_count, :, :, np.newaxis] * ground_truth.pixels[:recon_image_count],
                reconstruction_modes.background[:recon_image_count],
            ], axis=2)
        ], axis=1),
        reconstruction_modes.turntable_pixels,
        tf.concat([
            generative.pixels[:recon_image_count],
            tf.zeros_like(generative.pixels[:recon_image_count]) if background_embedding_dimensionality is None else generative.background[:recon_image_count]
        ], axis=1),
        generative.turntable_pixels
    ], axis=2), [recon_image_count * 2 * crop_height, -1, 3])
    encoded_image = tf.image.encode_jpeg(tf.cast(overview_images * 255., tf.uint8), quality=80)
    image_filename = tf.constant(all_images_path + '/', dtype=tf.string) + tf.as_string(global_step, width=6, fill='0') + '.jpg'
    write_images_op = tf.group(
        tf.write_file(image_filename, encoded_image),
        tf.write_file(latest_image_filename, encoded_image)
    )

    assert eval_images_count % images_per_batch == 0
    eval_batch_index = tf.placeholder(tf.int32, [])
    gen_pixels_u8 = tf.unstack(tf.cast(generative.pixels * 255., tf.uint8))
    gt_pixels_u8 = tf.unstack(tf.cast(ground_truth.pixels * 255., tf.uint8))
    eval_gen_images_path = all_images_path + '/eval-' + tf.as_string(global_step, width=6, fill='0')
    write_eval_gen_images_op = tf.group(*[
        tf.write_file(
            eval_gen_images_path + '/' + tf.as_string(eval_batch_index * images_per_batch + iib, width=5, fill='0') + '.jpg',
            tf.image.encode_jpeg(gen_pixels_u8[iib])
        )
        for iib in range(images_per_batch)
    ])
    write_eval_gt_images_op = tf.group(*[
        tf.write_file(
            all_images_path + '/eval-gt/' + tf.as_string(eval_batch_index * images_per_batch + iib, width=5, fill='0') + '.jpg',
            tf.image.encode_jpeg(gt_pixels_u8[iib])
        )
        for iib in range(images_per_batch)
    ])

    saver = tf.train.Saver()
    def save_checkpoint():
        checkpoint_path = saver.save(session, all_images_path + '/checkpoints/ckpt', global_step, write_state=False)
        print('checkpoint saved to ' + checkpoint_path)

    session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), inter_op_parallelism_threads=24))
    with session.as_default():

        restore_timestamp_and_step = hyper('', 'restore', str)
        if restore_timestamp_and_step != '':

            restore_timestamp, restore_step = restore_timestamp_and_step
            checkpoint_path = output_path + '/images/' + dataset_string + '/' + param_string
            if restore_timestamp == 'latest':
                timestamps = os.listdir(checkpoint_path)
                restore_timestamp = sorted(timestamps)[-2]  # most recent will be ourself!
            checkpoint_path += '/' + restore_timestamp + '/checkpoints'
            if restore_step == 'latest':
                checkpoint_files = os.listdir(checkpoint_path)
                if len(checkpoint_files) == 0:
                    raise RuntimeError('no checkpoint available to restore')
                checkpoint_filename = max(checkpoint_files, key=lambda filename: int(filename[5 : filename.index('.')]))
                checkpoint_filename = checkpoint_filename[:checkpoint_filename.index('.')]
            else:
                checkpoint_filename = 'ckpt-{}'.format(restore_step)
            print('restoring checkpoint ' + checkpoint_filename + ' of run ' + restore_timestamp)
            saver.restore(session, os.path.join(checkpoint_path, checkpoint_filename))

        else:

            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            tf.tables_initializer().run()

        eval_only = hyper(False, 'eval-only', bool)

        try:

            for batch_index in tqdm(range(eval_images_count // images_per_batch)):
                session.run(write_eval_gt_images_op, feed_dict={eval_batch_index: batch_index, is_training: False})

            while True:

                print('iteration {}'.format(global_step.eval()))

                try:

                    if not eval_only:
                        batch_loss, batch_step, _ = session.run(
                            [loss, global_step, train_op],
                            feed_dict={is_training: True, use_gt_background: False}
                        )
                        if not np.isfinite(batch_loss):
                            raise RuntimeError('non-finite loss')

                    if eval_only or batch_step % write_images_frequency == 0:
                        _, batch_summaries = session.run([write_images_op, val_summaries], feed_dict={is_training: False, use_gt_background: False})
                        if not eval_only:
                            writer.add_summary(batch_summaries, global_step=batch_step)

                    if eval_only or (write_eval_images_frequency is not None and batch_step % write_eval_images_frequency == 0):
                        save_checkpoint()
                        all_mesh_vertices = []
                        all_mesh_face_colours = []
                        for batch_index in tqdm(range(eval_images_count // images_per_batch)):
                            try:
                                [_, generated_vertices, generated_face_colours] = session.run(
                                    [write_eval_gen_images_op, generative.vertices_object, generative.face_colours],
                                    feed_dict={eval_batch_index: batch_index, is_training: False, use_gt_background: True}
                                )
                                if len(all_mesh_vertices) < eval_meshes_count:
                                    all_mesh_vertices.extend(generated_vertices)
                                    all_mesh_face_colours.extend(generated_face_colours)
                            except tf.errors.InvalidArgumentError as e:
                                print(e)
                        with open(eval_gen_images_path.eval().decode('ascii') + '/meshes.pickle', 'wb') as f:
                            pickle.dump([all_mesh_vertices, all_mesh_face_colours, Generative.vertex_anchors_and_faces[1]], f)
                        reconstructed_vertices, generated_vertices = session.run([reconstruction_modes.vertices_object, generative.vertices_object], feed_dict={is_training: False})
                        for mesh_index in range(recon_image_count):
                            write_obj('{}/meshes/recon-{:02}.obj'.format(all_images_path, mesh_index), reconstructed_vertices[mesh_index], Generative.vertex_anchors_and_faces[1])
                            write_obj('{}/meshes/gen-{:02}.obj'.format(all_images_path, mesh_index), generated_vertices[mesh_index], Generative.vertex_anchors_and_faces[1])

                    if eval_only:
                        break

                except tf.errors.InvalidArgumentError as e:
                    print(e)

        except tf.errors.OutOfRangeError:
            print('reached epoch limit')

        except KeyboardInterrupt:
            pass

        save_checkpoint()


if __name__ == '__main__':
    main()
