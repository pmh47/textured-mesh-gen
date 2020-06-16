
import numpy as np
import tensorflow as tf
import dirt.projection


def get_vanishing_point(first_line, second_line):

    # Based on apartment-sketching; may not be optimal here!
    # Derived using Mathematica's Eliminate & Solve on the vector equations
    ((a1x, a1y), (a2x, a2y)) = first_line
    ((b1x, b1y), (b2x, b2y)) = second_line
    epsilon = 1.e-9
    if a1y == a2y and b1y == b2y:
        return None
    if a1x == a2x:
        if b1x == b2x:
            return None
        # The elimination we use below isn't valid in this case
        beta = (a1x - b1x) / (b2x - b1x)
        alpha = (beta * (b2y - b1y) + b1y - a1y) / (a2y - a1y)
    else:
        beta = (-a2y * b1x + a1y * (-a2x + b1x) + a1x * (a2y - b1y) + a2x * b1y) / ((a1y - a2y) * (b1x - b2x) - (a1x - a2x) * (b1y - b2y) + epsilon)
        alpha = (beta * (b2x - b1x) + b1x - a1x) / (a2x - a1x + epsilon)
    return np.float32([a1x + alpha * (a2x - a1x), a1y + alpha * (a2y - a1y)])


def get_nearest_point(segment, point):

    # Based on apartment-sketching!
    # Returns the nearest point on the infinite line through segment[0] & segment[1], to point
    start, end = segment
    normaliser = np.linalg.norm(end - start)
    direction = (end - start) / normaliser
    alpha = np.dot(point - start, direction) / normaliser
    return start + alpha * (end - start)


def get_camera_matrices_from_vanishing_points(first_vp, second_vp, image_size):

    # See https://annals-csis.org/proceedings/2012/pliks/110.pdf and http://ksimek.github.io/2012/08/14/decompose/ [1]
    # first_vp & second_vp are batched x,y coordinates of pairs of parallel lines, in pixel space with the top-left corner as the origin
    # image_size is indexed by x/y

    # This is a 'traditional' calibration, not following OpenGL conventions, except that the camera is assumed to look along
    # the *negative* z-axis, i.e. it's a right-handed coordinate-system
    # The +ve world-x-axis will point from the camera-centre towards V1; the -ve world z-axis will point towards V2

    # The resulting matrices are assumed to *left*-multiply vectors, i.e. are indexed as out, in

    # Translate the vanishing points to be relative to the image centre, with y increasing as we move up
    V1 = (first_vp - image_size / 2.) * [1., -1.]
    V2 = (second_vp - image_size / 2.) * [1., -1.]

    # Calculate the focal length
    Vi = get_nearest_point([V1, V2], (0, 0))  # in centred image space, the nearest point to the image-centre (0, 0) that lies on the line between the two vanishing pionts
    Oi_Vi_sq = Vi[0] * Vi[0] + Vi[1] * Vi[1]
    Oc_Vi_sq = np.linalg.norm(Vi - V1) * np.linalg.norm(V2 - Vi)
    if Oc_Vi_sq < Oi_Vi_sq:
        raise ValueError
    f = np.sqrt(Oc_Vi_sq - Oi_Vi_sq)  # focal length measured in pixels
    # print 'estimated fov = {:.1f}deg'.format(2 * np.arctan(0.5 * image_size[0] / f) * 180 / np.pi)
    K = np.diag([f, f, 1.])

    # Calculate the world --> camera rotation matrix, which is made of the direction vectors Xc, Yc & Zc in camera space of the world axes
    Oc_V1 = np.concatenate([V1, [-f]])  # negation of f is because camera looks along negative-z
    Oc_V2 = np.concatenate([V2, [-f]])
    Xc = Oc_V1 / np.linalg.norm(Oc_V1)  # thus, physical lines pointing at V1 are parallel with the world-x-axis...
    Zc = Oc_V2 / np.linalg.norm(Oc_V2)  # ...and physical lines pointing at V2 are parallel with the world-z-axis
    if Zc[0] < 0:
        # Make sure the +ve z-axis (perpendicular to road) points right-ish in image space -- which it won't
        # do naturally if V2 is to the left of the image
        Zc = -Zc
    Yc = np.cross(Zc, Xc)
    assert Yc[1] > 0  # i.e. require that 'up is up'
    R = np.stack([Xc, Yc, Zc], axis=1)

    return K, R


def convert_camera_to_gl(K, R, image_size, near, far):

    # This assumes K, R are given wrt a left-handed camera (Hartley-Zisserman / OpenCV / etc.), i.e. the camera looks along the positive-z axis
    # See http://ksimek.github.io/2012/08/14/decompose/ [1] and http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    # It also assumes they left-multiply vectors, but the output matrices are transposed so they right-multiply, as in our other code
    # If image_size is None, then the x and y coordinates are left in pixel units instead of ndc

    perspective_matrix = np.asarray([
        [K[0, 0], K[0, 1], -K[0, 2], 0],
        [K[1, 0], K[1, 1], -K[1, 2], 0],
        [0, 0, near + far, near * far],
        [0, 0, -K[2, 2], 0]
    ])  # negation of K (of which only K[2, 2] is non-zero) is required because OpenGL clip coordinates have the camera looking along positive-z, in contrast to view coordinates
    ndc_matrix = np.asarray([
        [2. / image_size[0] if image_size is not None else 1., 0, 0, 0],
        [0, 2. / image_size[1] if image_size is not None else 1., 0, 0],
        [0, 0, -2. / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1]
    ])
    projection_matrix = np.dot(ndc_matrix, perspective_matrix)

    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R
    if np.linalg.det(view_matrix) < 0:  # step four of [1] sec. 4
        print('warning: view_matrix has negative determinant')
        view_matrix = -view_matrix

    return np.transpose(projection_matrix), np.transpose(view_matrix)  # transpose as we assume matrices right-multiply vectors everywhere else


def unproject_onto_ground(pixel_locations, clip_to_world_matrix, image_size):

    # This unprojects the given image-space locations onto the world-space y = 0 plane
    # pixel_locations is indexed by *, x/y
    # clip_to_world_matrix is indexed by x/y/z/w (in), x/y/z/w (out)
    # image_size is indexed by x/y
    # result is indexed by *, x/y/z

    pixel_ray_starts_world, pixel_ray_deltas_world = dirt.projection.unproject_pixels_to_rays(pixel_locations, clip_to_world_matrix, image_size)
    lambda_ground = -pixel_ray_starts_world[..., 1:2] / pixel_ray_deltas_world[..., 1:2]  # indexed by *, singleton
    return pixel_ray_starts_world + lambda_ground * pixel_ray_deltas_world

