
#include <igl/opengl/glfw/Viewer.h>

void view(Eigen::MatrixXf const &vertices, Eigen::MatrixXi const &faces)
{
	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(vertices.cast<double>(), faces);
	viewer.data().set_face_based(true);
	viewer.launch();
}

