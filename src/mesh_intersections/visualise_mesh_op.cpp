
#include <algorithm>

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "mesh_intersections.h"

using namespace tensorflow;

REGISTER_OP("VisualiseMesh")
	.Input("vertices: float32")
	.Input("faces: int32")
	.SetShapeFn( [] (::tensorflow::shape_inference::InferenceContext *c) {
		return Status::OK();
	} );

class VisualiseMeshOp : public OpKernel
{
public:

	explicit VisualiseMeshOp(OpKernelConstruction* context) : OpKernel(context)
	{
	}

	void Compute(OpKernelContext* context) override
	{
		Tensor const &vertices_tensor = context->input(0);
		OP_REQUIRES(
			context,
			vertices_tensor.shape().dims() == 2 && vertices_tensor.shape().dim_size(1) == 3,
			errors::InvalidArgument("VisualiseMesh expects vertices to be 2D, with final dimension of size three")
		);
		auto const vertices = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const>(
			vertices_tensor.matrix<float>().data(),
			vertices_tensor.shape().dim_size(0),
			vertices_tensor.shape().dim_size(1)
		);
		auto const vertex_count = vertices.rows();

		Tensor const &faces_tensor = context->input(1);
		OP_REQUIRES(
			context,
			faces_tensor.shape().dims() == 2 && faces_tensor.shape().dim_size(1) == 3,
			errors::InvalidArgument("VisualiseMesh expects faces to be 2D, with final dimension of size three")
		);
		auto const faces =  Eigen::Map<Eigen::Matrix<int32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const>(
			faces_tensor.matrix<int32>().data(),
			faces_tensor.shape().dim_size(0),
			faces_tensor.shape().dim_size(1)
		);

		view(vertices, faces);
	}
};

REGISTER_KERNEL_BUILDER(Name("VisualiseMesh").Device(DEVICE_CPU), VisualiseMeshOp);

