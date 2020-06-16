
#include <vector>
#include <map>
#include <iostream>
#include <chrono>
#include <cassert>

#include <boost/optional.hpp>
#include <boost/variant.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/hawick_circuits.hpp>

#include <Eigen/Core>  // ensure that this is Tensorflow's version of Eigen, not libIGL's -- see CMakeLists for info

#include <igl/opengl/glfw/Viewer.h>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Triangle_2.h>
#include <CGAL/intersections.h>
#include <CGAL/Barycentric_coordinates_2/Triangle_coordinates_2.h>

#include <gurobi_c++.h>

#include "mesh_intersections.h"


//#define CHECK_FOR_CYCLES  // if set, use boost::hawick_cycles to check the face-graph is acyclic
//#define CHECK_LINEAR_SOLVE  // if set, check the result of re-solving the linear system defined by the active constraints using Eigen
//#define RETURN_LINEAR_SOLVE  // if set, return that Eigen solution instead of the Gurobi solution
//#define VISUALISE_ACTIVE_CONSTRAINTS  // if set, display active constraint set after solving with gurobi
//#define MEASURE_TIMES  // if set, measure and report cumulative times for building the graph / LP and solving


typedef CGAL::Epeck CGALKernel;
typedef CGAL::Point_2<CGALKernel> Point2D;
typedef CGAL::Line_2<CGALKernel> Line2D;
typedef CGAL::Segment_2<CGALKernel> Segment2D;
typedef CGAL::Triangle_2<CGALKernel> Triangle2D;


struct IntersectionVertex
{
	float first_z, second_z;
	std::array<float, 3> first_barycentric, second_barycentric;  // corresponding to near/far triangle respectively

	void flip() {
		std::swap(first_z, second_z);
		std::swap(first_barycentric, second_barycentric);
	}
};

float interpolate_barycentric(std::array<float, 3> const &barycentric, float const v0, float const v1, float const v2)
{
	return v0 * barycentric[0] + v1 * barycentric[1] + v2 * barycentric[2];
}

float get_interpolated_z(std::array<float, 3> const &barycentric, Eigen::Vector3i const &face, Eigen::MatrixXf const &projected_vertices)
{
	auto const z_v0 = projected_vertices(face[0], 2);
	auto const z_v1 = projected_vertices(face[1], 2);
	auto const z_v2 = projected_vertices(face[2], 2);

	return interpolate_barycentric(barycentric, z_v0, z_v1, z_v2);
}

Eigen::Vector3f get_position_from_barycentric(std::array<float, 3> const &barycentric, Eigen::Vector3i const &face, Eigen::MatrixXf const &vertices)
{
	Eigen::Vector3f const v0 = vertices.row(face[0]);
	Eigen::Vector3f const v1 = vertices.row(face[1]);
	Eigen::Vector3f const v2 = vertices.row(face[2]);
	return v0 * barycentric[0] + v1 * barycentric[1] + v2 * barycentric[2];
}

typedef boost::adjacency_list<
    boost::vecS,
    boost::vecS,
    boost::directedS,
    boost::no_property,
    std::vector<IntersectionVertex>
> FaceGraph;

typedef std::array<float, 3> Barycentric;

struct TriangleVertex
{
	Eigen::Vector3f position;
	Point2D const &position_cgal;
	unsigned char index_in_face;

	Barycentric get_barycentric() const {
		Barycentric result {0.f, 0.f, 0.f};
		result[index_in_face] = 1.f;
		return result;
	}

	Eigen::Vector2d position_2d() const {
		return Eigen::Vector2d(position[0], position[1]);
	}
};

typedef std::array<TriangleVertex, 3> TriangleVertices;

struct BBox
{
	float min_x, min_y, max_x, max_y;

	explicit BBox(TriangleVertices const &vertices) :
		min_x(std::min(std::min(vertices[0].position[0], vertices[1].position[0]), vertices[2].position[0])),
		min_y(std::min(std::min(vertices[0].position[1], vertices[1].position[1]), vertices[2].position[1])),
		max_x(std::max(std::max(vertices[0].position[0], vertices[1].position[0]), vertices[2].position[0])),
		max_y(std::max(std::max(vertices[0].position[1], vertices[1].position[1]), vertices[2].position[1]))
	{
	}

	bool almost_intersects(BBox const &other, float const epsilon) const {
		if (std::max(min_x, other.min_x) > std::min(max_x, other.max_x) + epsilon)
			return false;
		else if (std::max(min_y, other.min_y) > std::min(max_y, other.max_y) + epsilon)
			return false;
		else
			return true;
	}
};

std::vector<IntersectionVertex> get_intersection_vertices(Eigen::Vector3i const &first_face, TriangleVertices const &first_vertices, Eigen::Vector3i const &second_face, TriangleVertices const &second_vertices)
{
	// Note that if the triangles touch at an edge or vertex, this function *may* return a one-/two-element
	// intersection, or *may* return an empty intersection; non-trivial intersections are always returned

	// Find the set of vertices that are shared by the two triangles; record the index-in-face for each triangle for each such vertex
	std::vector<std::pair<int, int>> shared_vertex_index_indices;
	shared_vertex_index_indices.reserve(3);
	for (int first_vertex_index_index = 0; first_vertex_index_index < 3; ++first_vertex_index_index) {
		for (int second_vertex_index_index = 0; second_vertex_index_index < 3; ++second_vertex_index_index) {
			if (first_face[first_vertex_index_index] == second_face[second_vertex_index_index]) {
				shared_vertex_index_indices.emplace_back(first_vertex_index_index, second_vertex_index_index);
				break;
			}
		}
	}

	// Find the complement of the above set wrt each triangle, i.e. the sets of non-shared vertex-indices-in-face for each triangle
	std::vector<int> first_nonshared_vertex_index_indices, second_nonshared_vertex_index_indices;
	first_nonshared_vertex_index_indices.reserve(3 - shared_vertex_index_indices.size());
	second_nonshared_vertex_index_indices.reserve(3 - shared_vertex_index_indices.size());
	for (int vertex_index_index = 0; vertex_index_index < 3; ++vertex_index_index) {
		bool shared_in_first = false, shared_in_second = false;
		for (auto [shared_index_index_in_first, shared_index_index_in_second] : shared_vertex_index_indices) {
			if (shared_index_index_in_first == vertex_index_index)
				shared_in_first = true;
			if (shared_index_index_in_second == vertex_index_index)
				shared_in_second = true;
		}
		if (!shared_in_first)
			first_nonshared_vertex_index_indices.push_back(vertex_index_index);
		if (!shared_in_second)
			second_nonshared_vertex_index_indices.push_back(vertex_index_index);
	}
	assert(shared_vertex_index_indices.size() + first_nonshared_vertex_index_indices.size() == 3);
	assert(shared_vertex_index_indices.size() + second_nonshared_vertex_index_indices.size() == 3);

	auto const triangle_to_cgal = [] (TriangleVertices const &triangle_vertices) {
		return Triangle2D{triangle_vertices[0].position_cgal, triangle_vertices[1].position_cgal, triangle_vertices[2].position_cgal};
	};

	auto const get_barycentric = [] (Eigen::Vector2d const &point_d, TriangleVertices const &triangle_vertices) {
		auto const cross2 = [] (Eigen::Vector2d const &first, Eigen::Vector2d const&second) {
			return first[0] * second[1] - first[1] * second[0];
		};
		Eigen::Vector2d const v0 = triangle_vertices[1].position_2d() - triangle_vertices[0].position_2d();
		Eigen::Vector2d const v1 = triangle_vertices[2].position_2d() - triangle_vertices[0].position_2d();
		Eigen::Vector2d const v2 = point_d - triangle_vertices[0].position_2d();
		auto const denominator = cross2(v0, v1);
		auto const v = cross2(v2, v1) / denominator;
		auto const w = cross2(v0, v2) / denominator;
		auto const u = 1.f - v - w;
		float const bounds_epsilon = 1.e-3;
		assert(-bounds_epsilon <= u && u <= 1.f + bounds_epsilon);
		assert(-bounds_epsilon <= v && v <= 1.f + bounds_epsilon);
		assert(-bounds_epsilon <= w && w <= 1.f + bounds_epsilon);
		return Barycentric{static_cast<float>(u), static_cast<float>(v), static_cast<float>(w)};
	};

	auto const get_barycentric_and_z = [&] (Eigen::Vector2d const &point_d, TriangleVertices const &triangle_vertices) {
		auto const barycentric = get_barycentric(point_d, triangle_vertices);
		auto const z = interpolate_barycentric(barycentric, triangle_vertices[0].position[2], triangle_vertices[1].position[2], triangle_vertices[2].position[2]);
		return std::make_pair(barycentric, z);
	};

	auto const with_barycentrics_and_zs = [&] (Point2D const &point) {
		Eigen::Vector2d const point_d(CGAL::to_double(point[0]), CGAL::to_double(point[1]));
		auto const [first_barycentric, first_z] = get_barycentric_and_z(point_d, first_vertices);
		auto const [second_barycentric, second_z] = get_barycentric_and_z(point_d, second_vertices);
		return IntersectionVertex{first_z, second_z, first_barycentric, second_barycentric};
	};

	if (shared_vertex_index_indices.size() == 0 || shared_vertex_index_indices.size() == 1) {

		auto const first_triangle = triangle_to_cgal(first_vertices);
		auto const second_triangle = triangle_to_cgal(second_vertices);

		auto const intersection = CGAL::intersection(first_triangle, second_triangle);
		if (intersection) {
			if (auto point = boost::get<Point2D>(&*intersection))
				return {};
			else if (auto segment = boost::get<Segment2D>(&*intersection))
				return {};
			else if (auto triangle = boost::get<Triangle2D>(&*intersection))
				return {
					with_barycentrics_and_zs(triangle->vertex(0)),
					with_barycentrics_and_zs(triangle->vertex(1)),
					with_barycentrics_and_zs(triangle->vertex(2))
				};
			else if (auto poly = boost::get<std::vector<Point2D>>(&*intersection)) {
				std::vector<IntersectionVertex> result;
				result.reserve(poly->size());
				for (auto const &point : *poly)
					result.push_back(with_barycentrics_and_zs(point));
				return result;
			} else
				assert(false);
		} else {
			return {};
		}

	} else if (shared_vertex_index_indices.size() == 2) {

		Segment2D const shared_edge{
			first_vertices[shared_vertex_index_indices[0].first].position_cgal,
			first_vertices[shared_vertex_index_indices[1].first].position_cgal
		};
		Point2D const first_unshared_vertex = first_vertices[first_nonshared_vertex_index_indices[0]].position_cgal;
		Point2D const second_unshared_vertex = second_vertices[second_nonshared_vertex_index_indices[0]].position_cgal;

		Line2D const shared_line = shared_edge.supporting_line();
		if (shared_line.oriented_side(first_unshared_vertex) == shared_line.oriented_side(second_unshared_vertex)) {

			// There is an area intersection; could have two edges crossing, or could have one non-shared vertex inside the other triangle

			std::array<IntersectionVertex, 2> const shared_edge_intersection_vertices {
				IntersectionVertex{
					first_vertices[shared_vertex_index_indices[0].first].position[2],
					second_vertices[shared_vertex_index_indices[0].second].position[2],
					first_vertices[shared_vertex_index_indices[0].first].get_barycentric(),
					second_vertices[shared_vertex_index_indices[0].second].get_barycentric()
				},
				IntersectionVertex{
					first_vertices[shared_vertex_index_indices[1].first].position[2],
					second_vertices[shared_vertex_index_indices[1].second].position[2],
					first_vertices[shared_vertex_index_indices[1].first].get_barycentric(),
					second_vertices[shared_vertex_index_indices[1].second].get_barycentric()
				}
			};

			auto const first_triangle = triangle_to_cgal(first_vertices);
			auto const second_triangle = triangle_to_cgal(second_vertices);

			if (!first_triangle.has_on_unbounded_side(second_unshared_vertex)) {  // not-unbounded accounts for possibility of lying exactly on the boundary
				auto const [first_barycentric, first_z] = get_barycentric_and_z(
					second_vertices[second_nonshared_vertex_index_indices[0]].position_2d(),
					first_vertices
				);
				return {
					shared_edge_intersection_vertices[0],
					shared_edge_intersection_vertices[1],
					IntersectionVertex{
						first_z,
						second_vertices[second_nonshared_vertex_index_indices[0]].position[2],
						first_barycentric,
						second_vertices[second_nonshared_vertex_index_indices[0]].get_barycentric()
					}
				};
			}
			if (!second_triangle.has_on_unbounded_side(first_unshared_vertex)) {
				auto const [second_barycentric, second_z] = get_barycentric_and_z(
					first_vertices[first_nonshared_vertex_index_indices[0]].position_2d(),
					second_vertices
				);
				return {
					shared_edge_intersection_vertices[0],
					shared_edge_intersection_vertices[1],
					IntersectionVertex{
						first_vertices[first_nonshared_vertex_index_indices[0]].position[2],
						second_z,
						first_vertices[first_nonshared_vertex_index_indices[0]].get_barycentric(),
						second_barycentric
					}
				};
			}

			std::array<Segment2D, 2> const first_nonshared_edges {
				Segment2D{shared_edge[0], first_unshared_vertex},
				Segment2D{shared_edge[1], first_unshared_vertex}
			};
			std::array<Segment2D, 2> const second_nonshared_edges {
				Segment2D{shared_edge[0], second_unshared_vertex},
				Segment2D{shared_edge[1], second_unshared_vertex}
			};

			auto const check_edge_intersection = [&] (int const first_nonshared_edge_index, int const second_nonshared_edge_index) -> std::optional<std::vector<IntersectionVertex>> {
				if (auto const intersection = CGAL::intersection(
					first_nonshared_edges[first_nonshared_edge_index],
					second_nonshared_edges[second_nonshared_edge_index]
				)) {
					if (auto const point = boost::get<Point2D>(&*intersection))
						return {{shared_edge_intersection_vertices[0], shared_edge_intersection_vertices[1], with_barycentrics_and_zs(*point)}};
					else
						assert(false);  // this implies the two segments are collinear, in which case the not-on-unbounded-side check above would have passed
				} else {
					return std::nullopt;
				}
			};

			if (auto const result = check_edge_intersection(0, 1))
				return *result;
			if (auto const result = check_edge_intersection(1, 0))
				return *result;

			assert(false);  // ...as the above cases should be exhaustive

		} else {
			// If the two non-shared vertices lie on different sides of the shared edge, then the triangles touch at the edge, but do not intersect
			return {};
		}

	} else {
		assert(false);
	}
}

FaceGraph get_face_graph_hybrid(Eigen::MatrixXf const &projected_vertices, Eigen::MatrixXi const &faces, Eigen::ArrayXf const &offset_magnitudes)
{
	// This returns a directed graph, whose nodes correspond to faces. Edges represent 'possible bumpings', i.e.
	// existence of an edge F --> G implies that F and G overlap in projection, and F is 'further back' than G, hence
	// may bump into it when moved 'forward'
	// It is 'hybrid' because it uses fast float32 pre-checks to avoid expensive infinite-precision CGAL intersection tests

	float const box_intersection_epsilon = 1.e-2f;

	FaceGraph graph(faces.rows());

	std::vector<Point2D> projected_vertices_cgal;
	projected_vertices_cgal.reserve(projected_vertices.size());
	for (int vertex_index = 0; vertex_index < projected_vertices.rows(); ++vertex_index)
		projected_vertices_cgal.push_back(Point2D{projected_vertices(vertex_index, 0), projected_vertices(vertex_index, 1)});

	for (int first_face_index = 0; first_face_index < faces.rows(); ++first_face_index) {

		Eigen::Vector3i const first_face = faces.row(first_face_index);
		TriangleVertices const first_face_vertices {
			TriangleVertex{projected_vertices.row(first_face[0]), projected_vertices_cgal[first_face[0]], 0},
			TriangleVertex{projected_vertices.row(first_face[1]), projected_vertices_cgal[first_face[1]], 1},
			TriangleVertex{projected_vertices.row(first_face[2]), projected_vertices_cgal[first_face[2]], 2}
		};
		BBox const first_bbox(first_face_vertices);

		for (int second_face_index = 0; second_face_index < first_face_index; ++second_face_index) {

			Eigen::Vector3i const second_face = faces.row(second_face_index);
			TriangleVertices const second_face_vertices {
				TriangleVertex{projected_vertices.row(second_face[0]), projected_vertices_cgal[second_face[0]], 0},
				TriangleVertex{projected_vertices.row(second_face[1]), projected_vertices_cgal[second_face[1]], 1},
				TriangleVertex{projected_vertices.row(second_face[2]), projected_vertices_cgal[second_face[2]], 2}
			};
			BBox const second_bbox(second_face_vertices);

			if (!first_bbox.almost_intersects(second_bbox, box_intersection_epsilon))
				continue;

			auto const intersection_vertices = get_intersection_vertices(first_face, first_face_vertices, second_face, second_face_vertices);
			if (intersection_vertices.size() < 3)
				continue;

			// This is necessary as some overlapping triangles may share one or two vertices, which then have zero z-difference
			float largest_magnitude_second_z_minus_first_z = 0.f;
			for (auto const &intersection_vertex : intersection_vertices) {
				auto const z_difference = intersection_vertex.second_z - intersection_vertex.first_z;
				if (std::abs(z_difference) > std::abs(largest_magnitude_second_z_minus_first_z))
					largest_magnitude_second_z_minus_first_z = z_difference;
			}
			assert(largest_magnitude_second_z_minus_first_z != 0.f);
			bool const first_nearer_than_second = largest_magnitude_second_z_minus_first_z > 0.f;
			float const epsilon = 1.e-4;
			for (auto const &intersection_vertex : intersection_vertices) {
				auto const z_difference = intersection_vertex.second_z - intersection_vertex.first_z;
				assert(std::abs(z_difference) < epsilon || (z_difference > 0) == first_nearer_than_second);
			}

			if (first_nearer_than_second)
				boost::add_edge(first_face_index, second_face_index, intersection_vertices, graph);
			else {
				auto flipped_intersection_vertices = intersection_vertices;
				for (auto &intersection_vertex : flipped_intersection_vertices)
					intersection_vertex.flip();
				boost::add_edge(second_face_index, first_face_index, flipped_intersection_vertices, graph);
			}
		}
	}

	return graph;
}

void visualise_ordering(std::vector<int> const &face_ordering, Eigen::MatrixXf const &projected_vertices, Eigen::MatrixXi const &faces)
{
	Eigen::MatrixXd vertices_d = projected_vertices.cast<double>();
	for (int vertex_index = 0; vertex_index < vertices_d.rows(); ++vertex_index) {
		if (vertices_d(vertex_index, 2) < -0.01)
			vertices_d(vertex_index, 2) -= 0.1;
		else if (vertices_d(vertex_index, 2) > 0.01)
			vertices_d(vertex_index, 2) += 0.1;
	}

	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(vertices_d, faces);
	viewer.data().set_face_based(true);

	Eigen::MatrixXd face_colours(faces.rows(), 3);
	face_colours.setConstant(0.5);
	viewer.data().set_colors(face_colours);

	int current_index_in_ordering = 0;
	viewer.callback_key_down = [&] (igl::opengl::glfw::Viewer &viewer, unsigned int key, int mod) {
		if (key == GLFW_KEY_RIGHT && current_index_in_ordering < face_ordering.size() - 1) {
			face_colours.row(face_ordering[current_index_in_ordering]).setConstant(1.);
			++current_index_in_ordering;
			viewer.data().set_colors(face_colours);
			return true;
		} else if (key == GLFW_KEY_LEFT && current_index_in_ordering > 0) {
			--current_index_in_ordering;
			face_colours.row(face_ordering[current_index_in_ordering]).setConstant(0.5);
			viewer.data().set_colors(face_colours);
			return true;
		}
		return false;
	};

	viewer.launch();
}

void visualise_active_constraints(Eigen::MatrixXf const &vertices, Eigen::MatrixXi const &faces, std::vector<int> const &active_vertex_indices, std::vector<std::pair<int, int>> const &active_face_indices)
{
	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(vertices.cast<double>(), faces);
	viewer.data().set_face_based(true);

	Eigen::MatrixXd face_colours(faces.rows(), 3);
	face_colours.setConstant(0.5);
	for (auto const &[near_face_index, far_face_index] : active_face_indices) {
		face_colours(near_face_index, 0) = 1.;
		face_colours(far_face_index, 2) = 1.;
	}
	viewer.data().set_colors(face_colours);

	Eigen::MatrixXd point_colours(vertices.rows(), 3);
	point_colours.setZero();
	point_colours.col(0).setConstant(1.);
	for (auto const &active_vertex_index : active_vertex_indices)
		point_colours.row(active_vertex_index) = Eigen::Vector3d(0., 1., 0.);
	viewer.data().set_points(vertices.cast<double>(), point_colours);
	viewer.data().point_size = 10;

	viewer.launch();
}

float const buffer_distance = .05f;  // this is how close triangles are allowed to come to another before they collide
float const initial_z_difference_tolerance = 1.e-4f;  // we allow faces to have passed each other by this far without asserting

std::pair<std::pair<int, int>, std::pair<int, int>> get_segment_vertex_indices(
	Eigen::Vector3i const &near_face,
	Eigen::Vector3i const &far_face,
	IntersectionVertex const &intersection_vertex
) {
	// This returns the indices of the vertices at the ends of the two lines that intersect at the given intersection-vertex

	// If we're at a corner of the near/far face, then return the two edges meeting there

	float const corner_epsilon = 1.e-5;
	float const edge_epsilon = 1.e-4;

	auto const check_corners = [&] (
		std::array<float, 3> const &barycentric,
		Eigen::Vector3i const &face
	) -> std::optional<std::pair<std::pair<int, int>, std::pair<int, int>>> {
		if (barycentric[0] >= 1. - corner_epsilon)
			return { {{face[0], face[1]}, {face[0], face[2]}} };
		else if (barycentric[1] >= 1. - corner_epsilon)
			return { {{face[1], face[0]}, {face[1], face[2]}} };
		else if (barycentric[2] >= 1. - corner_epsilon)
			return { {{face[2], face[0]}, {face[2], face[1]}} };
		else
			return {};
	};

	if (auto corner_result = check_corners(intersection_vertex.first_barycentric, near_face))
		return *corner_result;
	if (auto corner_result = check_corners(intersection_vertex.second_barycentric, far_face))
		return *corner_result;

	// We're not at a corner of either triangle, but we are (necessarily) on an edge of each, and there are zero
	// barycentrics (one per triangle) opposite these edges

	auto const get_edge_opposite_zero_barycentric = [&] (std::array<float, 3> const &barycentric, Eigen::Vector3i const &face) -> std::pair<int, int> {
		if (barycentric[0] <= edge_epsilon) {
			assert(barycentric[1] > edge_epsilon && barycentric[2] > edge_epsilon);
			return {face[1], face[2]};
		}
		else if (barycentric[1] <= edge_epsilon) {
			assert(barycentric[0] > edge_epsilon && barycentric[2] > edge_epsilon);
			return {face[0], face[2]};
		}
		else if (barycentric[2] <= edge_epsilon) {
			assert(barycentric[0] > edge_epsilon && barycentric[1] > edge_epsilon);
			return {face[0], face[1]};
		} else {
			std::ostringstream message_ss;
			message_ss << "get_segment_vertex_indices::get_edge_opposite_zero_barycentric did not find any barycentric less than " << edge_epsilon
				<< "; values: " << barycentric[0] << ", " << barycentric[1] << ", " << barycentric[2];
			throw std::runtime_error(message_ss.str());
		}
	};

	return {
		get_edge_opposite_zero_barycentric(intersection_vertex.first_barycentric, near_face),
		get_edge_opposite_zero_barycentric(intersection_vertex.second_barycentric, far_face)
	};
}

OffsetMagnitudesAndActiveConstraints solve_for_offsets_gurobi(
	Eigen::MatrixXf const &projected_vertices,
	Eigen::MatrixXi const &faces,
	FaceGraph const &face_graph,
	Eigen::ArrayXf const &initial_offset_magnitudes
) {
	static GRBEnv gurobi_env;

#ifdef MEASURE_TIMES
	auto start = std::chrono::high_resolution_clock::now();
#endif

	GRBModel model(gurobi_env);
#ifdef NDEBUG
	model.set(GRB_IntParam_LogToConsole, 0);
#endif
	model.set(GRB_IntParam_Threads, 1);
	model.set(GRB_DoubleParam_TimeLimit, 30.);
	assert(model.get(GRB_IntAttr_ModelSense) ==  GRB_MINIMIZE);

	std::vector<GRBVar> offset_variables;
	std::vector<GRBConstr> offset_greater_than_initial_constraints;
	offset_variables.reserve(projected_vertices.rows());
	for (int vertex_index = 0; vertex_index < projected_vertices.rows(); ++vertex_index) {
		// This defines the objective as the (unweighted) sum of all offset-magnitudes, and the lower-bound on each as the original value
		offset_variables.push_back(model.addVar(0., std::numeric_limits<double>::infinity(), 1., GRB_CONTINUOUS));
		offset_greater_than_initial_constraints.push_back(model.addConstr(offset_variables.back() >= initial_offset_magnitudes[vertex_index]));
	}

#ifdef CHECK_LINEAR_SOLVE
	// This linear system has one row per constraint, and one column per variable (i.e. the offset-magnitudes)
	// We first collect all constraints (which do not have an equality solution), then later extract the active ones
	// Here, we construct the LHS and RHS explicitly, but for tensorflow, we return a set of face/vertex indices
	// describing how to construct them
	std::vector<Eigen::VectorXf> all_push_constraint_linear_matrix_rows;
	std::vector<float> all_push_constraint_linear_vector_elements;
#endif

	std::vector<std::tuple<GRBConstr, int, int, IntersectionVertex const *>> face_pushing_constraints_and_relevant_face_indices_and_intersection_vertex_ptrs;
	for (auto [edge_it, end_edge_it] = boost::edges(face_graph); edge_it != end_edge_it; ++edge_it) {
		int const near_face_index = boost::source(*edge_it, face_graph);
		int const far_face_index = boost::target(*edge_it, face_graph);
		Eigen::Vector3i const near_face = faces.row(near_face_index);
		Eigen::Vector3i const far_face = faces.row(far_face_index);
		for (auto const &intersection_vertex : face_graph[*edge_it]) {

			auto const near_initial_z = get_interpolated_z(
				intersection_vertex.first_barycentric,
				near_face,
				projected_vertices
			);
			auto const far_initial_z = get_interpolated_z(
				intersection_vertex.second_barycentric,
				far_face,
				projected_vertices
			);

			auto const initial_z_difference = far_initial_z - near_initial_z;
			assert(initial_z_difference >= -initial_z_difference_tolerance);
			auto const required_z_difference = std::min(initial_z_difference, buffer_distance);

			Eigen::VectorXf linear_matrix_row;
			linear_matrix_row.setZero(projected_vertices.rows());
			for (int index_in_face = 0; index_in_face < 3; ++index_in_face) {
				linear_matrix_row[near_face[index_in_face]] += intersection_vertex.first_barycentric[index_in_face];
				linear_matrix_row[far_face[index_in_face]] -= intersection_vertex.second_barycentric[index_in_face];
			}
			float const linear_vector_element = far_initial_z - near_initial_z - required_z_difference;

			if (linear_matrix_row.cwiseAbs().maxCoeff() < 1.e-9 && linear_vector_element == 0.) {
				// Such a trivial constraint arises when the intersection-vertex is exactly at a vertex shared between the near and far faces
				continue;
			} else if (linear_matrix_row.cwiseAbs().maxCoeff() < 1.e-5) {
				std::cout << "input constraint max-abs-coeff = " << linear_matrix_row.cwiseAbs().maxCoeff() << "; rhs = " << linear_vector_element << std::endl;
			}

#ifdef CHECK_LINEAR_SOLVE
			all_push_constraint_linear_matrix_rows.push_back(linear_matrix_row);
			all_push_constraint_linear_vector_elements.push_back(linear_vector_element);
#endif

			GRBLinExpr const near_offset_magnitude =
				intersection_vertex.first_barycentric[0] * offset_variables[near_face[0]] +
				intersection_vertex.first_barycentric[1] * offset_variables[near_face[1]] +
				intersection_vertex.first_barycentric[2] * offset_variables[near_face[2]];
			GRBLinExpr const far_offset_magnitude =
				intersection_vertex.second_barycentric[0] * offset_variables[far_face[0]] +
				intersection_vertex.second_barycentric[1] * offset_variables[far_face[1]] +
				intersection_vertex.second_barycentric[2] * offset_variables[far_face[2]];

			GRBConstr const constraint = model.addConstr(near_initial_z + near_offset_magnitude <= far_initial_z + far_offset_magnitude - required_z_difference);
			face_pushing_constraints_and_relevant_face_indices_and_intersection_vertex_ptrs.push_back({constraint, near_face_index, far_face_index, &intersection_vertex});
		}
	}

#ifdef MEASURE_TIMES
	static long long total_build_ms = 0;
	auto const total_build_ms_incremented = __sync_add_and_fetch(&total_build_ms, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());
	std::cout << "gurobi build -> total " << total_build_ms_incremented << "ms\n";
	start = std::chrono::high_resolution_clock::now();
#endif

	model.optimize();
	auto const optimisation_status = model.get(GRB_IntAttr_Status);
	if (optimisation_status != GRB_OPTIMAL) {
		std::cout << "WARNING: optimisation not solved; status = " << optimisation_status << "; assuming all lower-bound constraints active" << std::endl;
		std::vector<int> all_vertex_indices(projected_vertices.rows());
		std::iota(all_vertex_indices.begin(), all_vertex_indices.end(), 0);
		return {initial_offset_magnitudes, all_vertex_indices, {}, {}};
	}

#ifdef MEASURE_TIMES
	static long long total_solve_ms = 0;
	auto const total_solve_ms_incremented = __sync_add_and_fetch(&total_solve_ms, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());
	std::cout << "gurobi solve -> total " << total_solve_ms_incremented << "ms\n";
#endif

#ifdef CHECK_LINEAR_SOLVE
	Eigen::MatrixXf active_constraint_linear_matrix(0, projected_vertices.rows());
	Eigen::VectorXf active_constraint_linear_vector(0, 1);
	auto append_constraint_to_linear_system = [&] (Eigen::VectorXf const &lhs, float const rhs) {
		active_constraint_linear_matrix.conservativeResize(active_constraint_linear_matrix.rows() + 1, active_constraint_linear_matrix.cols());
		active_constraint_linear_vector.conservativeResize(active_constraint_linear_vector.rows() + 1);
		active_constraint_linear_matrix.row(active_constraint_linear_matrix.rows() - 1) = lhs;
		active_constraint_linear_vector[active_constraint_linear_vector.rows() - 1] = rhs;
	};
#endif

	double const slack_epsilon = 1.e-5;  // note that Gurobi's default feasibility tolerance is 1.e-6

	std::unique_ptr<double const> const gurobi_offset_magnitudes_ptr(model.get(GRB_DoubleAttr_X, &offset_variables[0], offset_variables.size()));
	Eigen::VectorXf const gurobi_solved_offset_magnitudes = Eigen::Map<Eigen::VectorXd const>(gurobi_offset_magnitudes_ptr.get(), offset_variables.size()).cast<float>();

	std::vector<int> active_bound_constraint_vertex_indices;
	for (int vertex_index = 0; vertex_index < projected_vertices.rows(); ++vertex_index) {
		double const slack = offset_greater_than_initial_constraints[vertex_index].get(GRB_DoubleAttr_Slack);
		assert(slack <= slack_epsilon);  // these slack variables are negative as the constraints are greater-than
		if (slack >= -slack_epsilon) {
			active_bound_constraint_vertex_indices.push_back(vertex_index);
#ifdef CHECK_LINEAR_SOLVE
			Eigen::VectorXf linear_matrix_row;
			linear_matrix_row.setZero(projected_vertices.rows());
			linear_matrix_row[vertex_index] = 1.f;
			append_constraint_to_linear_system(linear_matrix_row, initial_offset_magnitudes[vertex_index]);
#endif
		}
	}

	std::vector<std::pair<int, int>> active_push_constraint_face_indices;  // pairs of near-face, far-face
	std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> active_push_constraint_vertex_indices;  // identifies the IntersectionVertex as the intersection point of two edges; pairs of first-edge, second-edge -- each a pair of start-vertex, end-vertex
	for (int constraint_index = 0; constraint_index < face_pushing_constraints_and_relevant_face_indices_and_intersection_vertex_ptrs.size(); ++constraint_index) {
		auto const &[constraint, near_face_index, far_face_index, intersection_vertex_ptr] = face_pushing_constraints_and_relevant_face_indices_and_intersection_vertex_ptrs[constraint_index];
		double const slack = constraint.get(GRB_DoubleAttr_Slack);
		assert(slack >= -slack_epsilon);  // these slack variables are positive as the constraints are less-than
		if (slack <= slack_epsilon) {
			active_push_constraint_face_indices.push_back({near_face_index, far_face_index});
			active_push_constraint_vertex_indices.push_back(get_segment_vertex_indices(faces.row(near_face_index), faces.row(far_face_index), *intersection_vertex_ptr));
#ifdef CHECK_LINEAR_SOLVE
			append_constraint_to_linear_system(
				all_push_constraint_linear_matrix_rows[constraint_index],
				all_push_constraint_linear_vector_elements[constraint_index]
			);
#endif
		}
	}
#ifndef NDEBUG
	std::cout <<
		active_bound_constraint_vertex_indices.size() << "/" << offset_greater_than_initial_constraints.size() << " active lower-bound constraints; " <<
		active_push_constraint_face_indices.size() << "/" << face_pushing_constraints_and_relevant_face_indices_and_intersection_vertex_ptrs.size() << " active face-push constraints; " <<
		projected_vertices.rows() << " variables" <<
		std::endl;
#endif
	assert(active_push_constraint_face_indices.size() == active_push_constraint_vertex_indices.size());
	int const active_constraint_count = active_bound_constraint_vertex_indices.size() + active_push_constraint_face_indices.size();
	assert(active_constraint_count >= projected_vertices.rows());

#ifdef VISUALISE_ACTIVE_CONSTRAINTS
	visualise_active_constraints(projected_vertices, faces, active_bound_constraint_vertex_indices, active_push_constraint_face_indices);
#endif

#ifdef CHECK_LINEAR_SOLVE

	auto const final_row_max = active_constraint_linear_matrix.row(active_constraint_linear_matrix.rows() - 1).cwiseAbs().maxCoeff();
	if (final_row_max < 1.e-2)
		std::cout << "final row max abs = " << final_row_max << std::endl;

	assert(active_constraint_linear_matrix.rows() == active_constraint_linear_vector.rows() && active_constraint_linear_matrix.rows() == active_constraint_count);
	float const lambda = 1.e-12f;
	Eigen::VectorXf const eigen_solved_offset_magnitudes = (
		active_constraint_linear_matrix.transpose() * active_constraint_linear_matrix + Eigen::MatrixXf::Identity(active_constraint_linear_matrix.cols(), active_constraint_linear_matrix.cols()) * lambda
	).llt().solve(active_constraint_linear_matrix.transpose() * active_constraint_linear_vector);
	for (int vertex_index = 0; vertex_index < projected_vertices.rows(); ++vertex_index) {
		float const gurobi_value = offset_variables[vertex_index].get(GRB_DoubleAttr_X);
		float const eigen_value = eigen_solved_offset_magnitudes[vertex_index];
		if (std::abs(gurobi_value - eigen_value) > 1.e-2) {
			std::cout << "vertex #" << vertex_index << ": gurobi = " << gurobi_value << ", eigen = " << eigen_value;
			if (gurobi_value != 0.)
				std::cout << "; relative error = " << std::abs(eigen_value - gurobi_value) / std::abs(gurobi_value) * 100 << "%";
			std::cout << std::endl;
		}
	}

	for (int active_constraint_index = 0; active_constraint_index < active_constraint_count; ++active_constraint_index) {
		auto const lhs_eigen = active_constraint_linear_matrix.row(active_constraint_index).dot(eigen_solved_offset_magnitudes);
		auto const lhs_gurobi = active_constraint_linear_matrix.row(active_constraint_index).dot(gurobi_solved_offset_magnitudes);
		auto const rhs = active_constraint_linear_vector[active_constraint_index];
		if (std::abs(lhs_eigen - rhs) > 1.e-4) {
			std::cout << "active constraint #" << active_constraint_index << ": lhs = " << lhs_eigen << ", rhs = " << rhs;
			if (rhs != 0.)
				std::cout << "; relative error = " << std::abs(lhs_eigen - rhs) / rhs * 100 << "%";
			std::cout << " (gurobi lhs = " << lhs_gurobi << ")" << std::endl;
		}
	}

#endif

#ifdef RETURN_LINEAR_SOLVE

	return {
		eigen_solved_offset_magnitudes.array(),
		active_bound_constraint_vertex_indices,
		active_push_constraint_face_indices,
		active_push_constraint_vertex_indices
	};

#else

	return {
		gurobi_solved_offset_magnitudes.array(),
		active_bound_constraint_vertex_indices,
		active_push_constraint_face_indices,
		active_push_constraint_vertex_indices
	};

#endif
}

struct CycleVisitor {
	template<class Cycle>
	void cycle(Cycle const &, FaceGraph const &) {
		assert(false);
	}
};

OffsetMagnitudesAndActiveConstraints get_pushed_offset_magnitudes_and_active_constraints(Eigen::MatrixXf const &projected_vertices, Eigen::MatrixXi const &faces, Eigen::ArrayXf const &initial_offset_magnitudes)
{
	assert(projected_vertices.rows() == initial_offset_magnitudes.size());
	for (int vertex_index = 0; vertex_index < projected_vertices.rows(); ++vertex_index)
		assert(initial_offset_magnitudes[vertex_index] >= 0.f);

#ifdef MEASURE_TIMES
	auto start = std::chrono::high_resolution_clock::now();
#endif

	auto const face_graph = get_face_graph_hybrid(projected_vertices, faces, initial_offset_magnitudes);

#ifdef MEASURE_TIMES
	static long long total_ms = 0;
	auto const total_ms_incremented = __sync_add_and_fetch(&total_ms, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());
	std::cout << "intersections & graph construction -> total " << total_ms_incremented << "ms\n";
#endif

#ifdef CHECK_FOR_CYCLES
	boost::hawick_circuits(face_graph, CycleVisitor());
#endif

	return solve_for_offsets_gurobi(projected_vertices, faces, face_graph,  initial_offset_magnitudes);
}

