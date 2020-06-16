
import os
import tensorflow as tf


_library = tf.load_op_library(os.path.dirname(__file__ ) + '/libpush_ops.so')

_buffer_distance = 0.05  # ** currently duplicated in C++; should share!


def _dot(a, b):
    assert a.get_shape()[-1] == b.get_shape()[-1] == 3  # should never broadcast on x/y/z dimension!
    broadcast_shape = tf.broadcast_static_shape(a.get_shape(), b.get_shape())
    a = tf.broadcast_to(a, broadcast_shape)
    b = tf.broadcast_to(b, broadcast_shape)
    return tf.reduce_sum(a * b, axis=-1)


def _cross(a, b):
    assert a.get_shape()[-1] == b.get_shape()[-1] == 3  # should never broadcast on x/y/z dimension!
    broadcast_shape = tf.broadcast_static_shape(a.get_shape(), b.get_shape())
    a = tf.broadcast_to(a, broadcast_shape)
    b = tf.broadcast_to(b, broadcast_shape)
    return tf.cross(a, b)


def _get_rotation_matrix(z_axis):

    # This returns a rotation matrix, that causes the direction given by z_axis to become aligned with the canonical z-axis
    # z_axis :: iib, x/y/z

    def get_perpendicular_axis(candidate_direction):
        return candidate_direction - z_axis * _dot(z_axis, candidate_direction)[:, None]

    x_aligned_x_axis = get_perpendicular_axis(tf.constant([1., 0., 0.], dtype=tf.float32))
    y_aligned_x_axis = get_perpendicular_axis(tf.constant([0., 1., 0.], dtype=tf.float32))

    x_axis = tf.linalg.l2_normalize(tf.where(
        tf.tile(tf.less(tf.linalg.norm(x_aligned_x_axis, axis=-1, keepdims=True), 1.e-2), [1, 3]),
        y_aligned_x_axis,
        x_aligned_x_axis
    ), axis=-1)  # :: iib, x/y/z

    y_axis = _cross(z_axis, x_axis)

    return tf.stack([x_axis, y_axis, z_axis], axis=2)  # :: iib, x/y/z (in), x/y/z (out)


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _get_2d_line_intersections(first_line_point, first_line_direction, second_line_point, second_line_direction):

    # This does not require *_direction to be normalised. It finds the intersection of infinite lines, not a segment
    # all inputs :: *, x/y, where * is arbitrary batch dimensions common to the different inputs

    assert first_line_point.get_shape()[-1] == first_line_direction.get_shape()[-1] == 2
    assert second_line_point.get_shape()[-1] == second_line_direction.get_shape()[-1] == 2

    t = _cross2(second_line_point - first_line_point, second_line_direction) / _cross2(first_line_direction, second_line_direction)  # :: *

    return first_line_point + t[..., None] * first_line_direction


def _get_barycentrics_2d(points, triangle_vertices):

    # points :: *, x/y
    # triangle_vertices :: *, vertex-in-face, x/y

    assert triangle_vertices.get_shape().ndims == points.get_shape().ndims + 1
    assert points.get_shape()[-1] == triangle_vertices.get_shape()[-1] == 2

    points = tf.cast(points, tf.float64)
    triangle_vertices = tf.cast(triangle_vertices, tf.float64)

    v0 = triangle_vertices[..., 1, :] - triangle_vertices[..., 0, :]
    v1 = triangle_vertices[..., 2, :] - triangle_vertices[..., 0, :]
    v2 = points - triangle_vertices[..., 0, :]
    denominator = _cross2(v0, v1)
    v = _cross2(v2, v1) / denominator
    w = _cross2(v0, v2) / denominator
    u = 1. - v - w

    return tf.stack([u, v, w], axis=-1)


def _get_pushed_offset_magnitudes(rotated_vertices, faces, initial_offset_magnitudes):

    # rotated_vertices :: iib, vertex, x/y/z
    # faces :: face, vertex-in-face
    # initial_offset_magnitudes :: iib, vertex

    # This assumes that the offsets are applied along the +ve z-axis

    batch_size = rotated_vertices.get_shape()[0].value
    vertex_count = rotated_vertices.get_shape()[1].value

    def do_for_iib(rotated_vertices_for_iib, initial_offset_magnitudes_for_iib):

        pusher_outputs = _library.push_offsets(rotated_vertices=rotated_vertices_for_iib, faces=faces, initial_offset_magnitudes=initial_offset_magnitudes_for_iib)  # :: vertex

        # We have now...
        # pusher_outputs.final_offset_magnitudes :: vertex
        # pusher_outputs.active_bound_constraint_vertex_indices :: lb-constraint
        # pusher_outputs.active_push_constraint_face_indices :: fp-constraint, near-/far-face
        # pusher_outputs.active_push_constraint_vertex_indices :: fp-constraint, first-/second-line, endpoint

        # Build the linear equations for the lower-bound constraints, which simply say the relevant offset-magnitude
        # should equal its initial value

        lb_lhs_matrix = tf.one_hot(pusher_outputs.active_bound_constraint_vertex_indices, vertex_count)  # :: lb-constraint, vertex
        lb_rhs_vector = tf.gather(initial_offset_magnitudes_for_iib, pusher_outputs.active_bound_constraint_vertex_indices)  # :: lb-constraint

        # For each face-push constraint, find the projected (2D) location of the intersection-vertex

        fp_line_vertices_2d = tf.gather(rotated_vertices_for_iib[:, :2], pusher_outputs.active_push_constraint_vertex_indices)  # :: fp-constraint, first-/second-line, endpoint, x/y
        fp_first_line_points_2d = fp_line_vertices_2d[:, 0, 0]
        fp_first_line_directions_2d = fp_line_vertices_2d[:, 0, 1] - fp_first_line_points_2d
        fp_second_line_points_2d = fp_line_vertices_2d[:, 1, 0]
        fp_second_line_directions_2d = fp_line_vertices_2d[:, 1, 1] - fp_second_line_points_2d

        intersection_vertices_2d = _get_2d_line_intersections(
            fp_first_line_points_2d, fp_first_line_directions_2d,
            fp_second_line_points_2d, fp_second_line_directions_2d
        )  # :: fp-constraint, x/y

        # For each face-push constraint, find the barycentric coordinates of the intersection-vertex
        # wrt both the near and far triangles

        fp_face_vertex_indices = tf.gather(faces, pusher_outputs.active_push_constraint_face_indices)  # :: fp-constraint, near-/far-face, vertex-in-face
        near_and_far_face_vertices = tf.gather(rotated_vertices_for_iib, fp_face_vertex_indices)  # :: fp-constraint, near-/far-face, vertex-in-face, x/y/z
        near_face_vertices, far_face_vertices = tf.unstack(near_and_far_face_vertices, axis=1)  #  :: fp-constraint, vertex-in-face, x/y/z
        near_face_barycentrics = _get_barycentrics_2d(intersection_vertices_2d, near_face_vertices[:, :, :2])  # :: fp-constraint, vertex-in-face
        far_face_barycentrics = _get_barycentrics_2d(intersection_vertices_2d, far_face_vertices[:, :, :2])

        # Use the barycentrics to find the initial z values of the near and far faces at the intersection-vertex

        near_initial_zs = tf.reduce_sum(near_face_barycentrics * tf.cast(near_face_vertices[:, :, 2], tf.float64), axis=-1)  # :: fp-constraint
        far_initial_zs = tf.reduce_sum(far_face_barycentrics * tf.cast(far_face_vertices[:, :, 2], tf.float64), axis=-1)
        initial_z_differences = far_initial_zs - near_initial_zs
        required_z_differences = tf.minimum(initial_z_differences, _buffer_distance)  # :: fp-constraint

        # Construct the linear equations for the face-push constraints

        fp_constraint_count = tf.shape(pusher_outputs.active_push_constraint_face_indices)[0]
        fp_lhs_matrix = tf.scatter_nd(
            indices=tf.stack([
                tf.tile(tf.range(fp_constraint_count)[:, None, None], [1, 2, 3]),
                fp_face_vertex_indices
            ], axis=-1),  # :: fp-constraint, near-/far-face, vertex-in-face, fp-constraint/vertex
            updates=tf.stack([
                near_face_barycentrics,
                -far_face_barycentrics
            ], axis=1),  # :: fp-constraint, near-/far-face, vertex-in-face
            shape=[fp_constraint_count, vertex_count]
        )  # :: fp-constraint, vertex

        fp_rhs_vector = far_initial_zs - near_initial_zs - required_z_differences  # :: fp-constraint

        # Solve the combined linear system for the new offset-magnitudes

        lhs_matrix = tf.concat([tf.cast(lb_lhs_matrix, tf.float64), fp_lhs_matrix], axis=0)  # :: constraint, vertex
        rhs_vector = tf.concat([tf.cast(lb_rhs_vector, tf.float64), fp_rhs_vector], axis=0)  # :: constraint

        pushed_offset_magnitudes = tf.linalg.lstsq(lhs_matrix, rhs_vector[:, None], l2_regularizer=1.e-12)[:, 0]  # vertex
        pushed_offset_magnitudes = tf.cast(pushed_offset_magnitudes, tf.float32)

        # Check for numerical issues

        solver_discrepancies = tf.abs(pusher_outputs.final_offset_magnitudes - pushed_offset_magnitudes)
        def rank_fn(A, b, g, t):
            import numpy as np
            r = np.linalg.matrix_rank(A)
            # with open('matrices_{}.py'.format(np.random.randint(1000)), 'w') as f:
            #     f.write('A = [\n')
            #     for ct in A:
            #         f.write('[' + ', '.join(map(str, ct)) + '],\n')
            #     f.write(']\n')
            #     f.write('b = [' + ', '.join(map(str, b)) + ']\n')
            #     f.write('g = [' + ', '.join(map(str, g)) + ']\n')
            #     f.write('t = [' + ', '.join(map(str, t)) + ']\n')
            return r
        pushed_offset_magnitudes = tf.cond(
            tf.reduce_max(solver_discrepancies) > 1.e-1,
            lambda: tf.Print(pushed_offset_magnitudes, [tf.reduce_max(solver_discrepancies), tf.reduce_mean(solver_discrepancies), tf.shape(lhs_matrix), tf.py_func(rank_fn, [lhs_matrix, rhs_vector, pusher_outputs.final_offset_magnitudes, pushed_offset_magnitudes], tf.int64, stateful=False)], 'warning: max / mean abs solver discrepancy & lhs shape & rank = '),
            lambda: pushed_offset_magnitudes
        )

        # Return the Gurobi solution, but back-propagate through the Cholesky solve

        return tf.stop_gradient(pusher_outputs.final_offset_magnitudes) + pushed_offset_magnitudes - tf.stop_gradient(pushed_offset_magnitudes)

    return tf.map_fn(lambda inputs: do_for_iib(*inputs), [rotated_vertices, initial_offset_magnitudes], dtype=tf.float32, parallel_iterations=32)


def apply_offsets_with_pushing(vertices, faces, offset_direction, initial_offset_magnitudes):

    # vertices :: iib, vertex, x/y/z
    # faces :: face, vertex-in-face
    # offset_direction :: iib, x/y/z
    # initial_offset_magnitudes :: iib, vertex

    # offset_direction should be a unit vector
    # initial_offset_magnitudes should all be positive

    with tf.control_dependencies([
        tf.check_numerics(vertices, 'check_numerics failed for vertices in apply_offsets_with_pushing'),
        tf.check_numerics(offset_direction, 'check_numerics failed for offset_direction in apply_offsets_with_pushing'),
        tf.check_numerics(initial_offset_magnitudes, 'check_numerics failed for initial_offset_magnitudes in apply_offsets_with_pushing')
    ]):
        vertices = tf.identity(vertices)

    rotation_matrix = _get_rotation_matrix(offset_direction)
    rotated_vertices = tf.matmul(vertices, rotation_matrix)  # :: iib, vertex, x/y/z

    with tf.device('/cpu:0'):
        pushed_offset_magnitudes = _get_pushed_offset_magnitudes(rotated_vertices, faces, initial_offset_magnitudes)

    shifted_rotated_vertices = rotated_vertices + tf.stack([
        tf.zeros_like(pushed_offset_magnitudes),
        tf.zeros_like(pushed_offset_magnitudes),
        pushed_offset_magnitudes
    ], axis=2)

    return tf.matmul(shifted_rotated_vertices, rotation_matrix, transpose_b=True)


def visualise_mesh(vertices, faces):

    _library.visualise_mesh(vertices=vertices, faces=faces)

