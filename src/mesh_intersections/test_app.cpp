
#include <igl/readOBJ.h>
#include <igl/per_face_normals.h>

#include "mesh_intersections.h"


void fix_normals(Eigen::MatrixXf const &vertices, Eigen::MatrixXi &faces)
{
	Eigen::MatrixXf face_normals;
	igl::per_face_normals(vertices, faces, face_normals);

	for (int face_index = 0; face_index < faces.rows(); ++face_index) {
		auto const face_centroid = (vertices.row(faces(face_index, 0)) + vertices.row(faces(face_index, 1)) + vertices.row(faces(face_index, 2))) / 3.f;
		if (face_normals.row(face_index).dot(face_centroid) < 0.)
			std::swap(faces(face_index, 0), faces(face_index, 2));
	}
}

Eigen::MatrixXf project_vertices(Eigen::MatrixXf const &vertices, Eigen::Vector3f const &z_axis)
{
	// Projects the given vertices onto a (random) plane perpendicular to the given 'z' axis

	Eigen::Vector3f x_axis(1.f, 0.f, 0.f);
	x_axis -= z_axis * z_axis.dot(x_axis);
	if (x_axis.norm() < 1.e-2) {  // i.e. require the (arbitrary) x-axis direction not to be too close to the z-axis
		x_axis = Eigen::Vector3f(0.f, 1.f, 0.f);
		x_axis -= z_axis * z_axis.dot(x_axis);
		assert(x_axis.norm() > 1.e-2);
	}
	x_axis = x_axis.normalized();

	Eigen::Vector3f const y_axis = z_axis.cross(x_axis);
	Eigen::Matrix3f transform_matrix;
	transform_matrix.col(0) = x_axis;
	transform_matrix.col(1) = y_axis;
	transform_matrix.col(2) = z_axis;

	return vertices * transform_matrix;
}

Eigen::MatrixXf get_transformed_vertices(Eigen::MatrixXf const &vertices, Eigen::Vector3f const &offset_direction, Eigen::ArrayXf const &offset_magnitudes)
{
	assert(vertices.rows() == offset_magnitudes.rows());
	Eigen::MatrixXf transformed_vertices(vertices.rows(), vertices.cols());
	for (int vertex_index = 0; vertex_index < vertices.rows(); ++vertex_index) {
		transformed_vertices(vertex_index, 0) = vertices(vertex_index, 0) + offset_direction[0] * offset_magnitudes[vertex_index];
		transformed_vertices(vertex_index, 1) = vertices(vertex_index, 1) + offset_direction[1] * offset_magnitudes[vertex_index];
		transformed_vertices(vertex_index, 2) = vertices(vertex_index, 2) + offset_direction[2] * offset_magnitudes[vertex_index];
	}
	return transformed_vertices;
}

Eigen::MatrixXf apply_pushing_deformation(Eigen::MatrixXf const &vertices, Eigen::MatrixXi const &faces, Eigen::Vector3f const &offset_direction, Eigen::ArrayXf const &initial_offset_magnitudes)
{
	// offset_magnitudes and offset_direction define a mesh deformation, in terms of a vector to be
	// added to each vertex
	// offset_direction is assumed to be a normalised direction vector (we assert if not)
	// offset_magnitudes is assumed to have all elements positive (or zero) (we assert if not)

	assert(std::abs(offset_direction.norm() - 1.f) < 1.e-4f);

	auto const projected_vertices = project_vertices(vertices, offset_direction);
//	view(projected_vertices, faces);

	auto const final_offset_magnitudes = get_pushed_offset_magnitudes_and_active_constraints(projected_vertices, faces, initial_offset_magnitudes).final_offset_magnitudes;

	return get_transformed_vertices(vertices, offset_direction, final_offset_magnitudes);
}

int main()
{
	auto const mesh_filename = "test-icosphere.obj";
	Eigen::MatrixXd vertices_d;
	Eigen::MatrixXi faces;
	igl::readOBJ(mesh_filename, vertices_d, faces);

	Eigen::MatrixXf vertices = vertices_d.cast<float>();
	vertices.col(0) *= 0.5f;
	fix_normals(vertices, faces);

	Eigen::ArrayXf offset_magnitudes(vertices.rows());
	offset_magnitudes.setZero();
	for (int vertex_index = 0; vertex_index < vertices.rows(); ++vertex_index) {
		auto vertex = vertices.row(vertex_index);
		if (vertices(vertex_index, 0) > 0.45) {
			offset_magnitudes[vertex_index] = 1.f;
		}
	}

	Eigen::Vector3f offset_direction(-1.f, 0.05f, 0.f);
	offset_direction /= offset_direction.norm();

	vertices = apply_pushing_deformation(vertices, faces, offset_direction, offset_magnitudes);
	view(vertices, faces);

	offset_magnitudes.setZero();
	for (int vertex_index = 0; vertex_index < vertices.rows(); ++vertex_index) {
		auto vertex = vertices.row(vertex_index);
		if (vertices(vertex_index, 2) > 0.9) {
			offset_magnitudes[vertex_index] = 1.f;
		}
	}

	offset_direction = Eigen::Vector3f(0.f, 0.01f, -1.f).normalized();
	vertices = apply_pushing_deformation(vertices, faces, offset_direction, offset_magnitudes);
	view(vertices, faces);

	return 0;
}

