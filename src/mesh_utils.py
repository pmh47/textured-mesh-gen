
from collections import namedtuple
import numpy as np

Mesh = namedtuple('Mesh', ['vertices', 'faces'])


def write_obj(filename, vertices, faces):

    with open(filename, 'w') as f:

        for vertex in vertices:
            f.write('v  {} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))

        for face in faces:
            f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))


def make_unit_cube():
    vertices = np.array(
        [[x, y, z] for z in [-0.5, 0.5] for y in [-0.5, 0.5] for x in [-0.5, 0.5]],
        dtype=np.float32
    )  # indexed by vertex-index, x/y/z
    quads = np.int32([
        [0, 1, 3, 2], [4, 5, 7, 6],  # front, back
        [0, 4, 5, 1], [2, 6, 7, 3],  # top, bottom
        [0, 2, 6, 4], [1, 5, 7, 3],  # left, right
    ])
    return vertices, quads


def quads_to_triangles(quads):
    # quads is indexed by quad-index, vertex-in-quad
    return sum([
        [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
        for quad in quads
    ], [])


def split_vertices_by_face(vertices, faces):
    # This assumes vertices and faces are 2D lists/arrays, with no batch dimension; it supports
    # arbitrary polygons (not just triangles)
    new_vertices = []
    new_faces = []
    for face in faces:
        new_faces.append([len(new_vertices) + vertex_in_face for vertex_in_face in range(len(face))])
        new_vertices.extend([vertices[vertex_index] for vertex_index in face])
    return np.asarray(new_vertices, dtype=np.float32), new_faces


def point_normals_outward(vertices, faces):

    # This flips face normals as required s.t. all faces point 'away' from the vertex-centroid of the mesh
    # Each faces is processed independently, without considering connectivity, so it is only suitable for convex-ish shapes

    # vertices :: vertex, x/y/z
    # faces :: face, vertex-in-face

    # This part is based on DIRT's vertex_normals_pre_split
    vertices_by_face = vertices[faces]  # :: face, vertex-in-face, x/y/z
    face_normals = np.cross(vertices_by_face[:, 1] - vertices_by_face[:, 0], vertices_by_face[:, 2] - vertices_by_face[:, 0])  # :: face, x/y/z
    face_normals /= (np.linalg.norm(face_normals, axis=-1, keepdims=True) + 1.e-12)

    mesh_centroid = np.mean(vertices, axis=0)  # :: x/y/z
    face_centroids = np.mean(vertices_by_face, axis=1)  # :: face, x/y/z
    needs_flipping = np.einsum('fv,fv->f', face_normals, face_centroids - mesh_centroid) < 0.

    flipped_faces = faces[:, ::-1]
    resulting_faces = np.where(np.tile(needs_flipping[:, None], [1, 3]), flipped_faces, faces)

    return resulting_faces


def get_adjacency_matrix(faces):

    vertex_count = np.max(faces) + 1
    A = np.zeros([vertex_count, vertex_count], np.uint8)
    for v1, v2, v3 in faces:
        A[v1, v2] = A[v2, v1] = 1
        A[v2, v3] = A[v3, v2] = 1
        A[v3, v1] = A[v1, v3] = 1
    return A


def get_creases(faces):

    # Result is indexed by crease-index, 1st endpoint / 2nd endpoint / 'left' other-vertex / 'right' other-vertex
    # There is one entry per edge in the mesh; pairs of endpoints correspond to the actual edges; 'left'/'right' vertices are the
    # other vertices that belong to the two face that including the relevant edge
    # Note that this assumes 'simple' topology, with exactly one or two faces touching each edge; edges with only one face
    # touching are not included in the result (as they do not constitute a crease)

    creases = []

    faces = np.asarray(faces)

    for first_face_index in range(len(faces)):
        for second_face_index in range(first_face_index):
            # If any two vertices of 1st face are in 2nd face too, then we've found a crease
            second_as_list = faces[second_face_index].tolist()
            indices_in_second_of_first = [
                second_as_list.index(first) if first in second_as_list else -1
                for first in faces[first_face_index]
            ]
            def other_in_second():
                if 0 not in indices_in_second_of_first:
                    return second_as_list[0]
                elif 1 not in indices_in_second_of_first:
                    return second_as_list[1]
                elif 2 not in indices_in_second_of_first:
                    return second_as_list[2]
                else:
                    assert False
            if indices_in_second_of_first[0] != -1 and indices_in_second_of_first[1] != -1:
                creases.append((faces[first_face_index, 0], faces[first_face_index, 1], faces[first_face_index, 2], other_in_second()))
            elif indices_in_second_of_first[1] != -1 and indices_in_second_of_first[2] != -1:
                creases.append((faces[first_face_index, 1], faces[first_face_index, 2], faces[first_face_index, 0], other_in_second()))
            elif indices_in_second_of_first[2] != -1 and indices_in_second_of_first[0] != -1:
                creases.append((faces[first_face_index, 2], faces[first_face_index, 0], faces[first_face_index, 1], other_in_second()))

    return np.int32(creases)


