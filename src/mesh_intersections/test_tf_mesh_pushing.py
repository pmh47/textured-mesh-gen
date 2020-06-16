
import meshzoo
import tensorflow as tf

from mesh_utils import point_normals_outward
from mesh_pushing import apply_offsets_with_pushing, visualise_mesh


def main():

    tf.enable_eager_execution()

    vertices, faces = meshzoo.iso_sphere(2)
    faces = point_normals_outward(vertices, faces)

    vertices[:, 0] *= 0.5

    vertices = tf.constant(vertices, dtype=tf.float32)
    faces = tf.constant(faces, dtype=tf.int32)

    visualise_mesh(vertices, faces)

    offset_direction = tf.linalg.l2_normalize([-1, 0.05, 0.])
    offset_magnitudes = tf.where(tf.greater(vertices[:, 0], 0.45), tf.ones([len(vertices)]), tf.zeros([len(vertices)]))
    vertices = apply_offsets_with_pushing(vertices[None], faces, offset_direction[None], offset_magnitudes[None])[0]
    visualise_mesh(vertices, faces)

    offset_direction = tf.linalg.l2_normalize([0., 0.01, -1.])
    offset_magnitudes = tf.where(tf.greater(vertices[:, 2], 0.9), tf.ones([len(vertices)]), tf.zeros([len(vertices)]))
    vertices = apply_offsets_with_pushing(vertices[None], faces, offset_direction[None], offset_magnitudes[None])[0]
    visualise_mesh(vertices, faces)


if __name__ == '__main__':

    main()

