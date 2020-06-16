#ifndef MESH_INTERSECTIONS_MESH_INTERSECTIONS_H
#define MESH_INTERSECTIONS_MESH_INTERSECTIONS_H

#include <Eigen/Core>

struct OffsetMagnitudesAndActiveConstraints
{
	Eigen::ArrayXf final_offset_magnitudes;
	std::vector<int> active_bound_constraint_vertex_indices;
	std::vector<std::pair<int, int>> active_push_constraint_face_indices;  // pairs of near-face, far-face
	std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> active_push_constraint_vertex_indices;  // pairs of first-edge, second-edge -- each a pair of start-vertex, end-vertex
};

OffsetMagnitudesAndActiveConstraints get_pushed_offset_magnitudes_and_active_constraints(Eigen::MatrixXf const &projected_vertices, Eigen::MatrixXi const &faces, Eigen::ArrayXf const &initial_offset_magnitudes);
void view(Eigen::MatrixXf const &vertices, Eigen::MatrixXi const &faces);

#endif //MESH_INTERSECTIONS_MESH_INTERSECTIONS_H
