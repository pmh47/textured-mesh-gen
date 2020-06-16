
#include <algorithm>

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <gurobi_c++.h>

#include "mesh_intersections.h"

using namespace tensorflow;

REGISTER_OP("PushOffsets")
	.Input("rotated_vertices: float32")
	.Input("faces: int32")
	.Input("initial_offset_magnitudes: float32")
	.Output("final_offset_magnitudes: float32")
	.Output("active_bound_constraint_vertex_indices: int32")
	.Output("active_push_constraint_face_indices: int32")
	.Output("active_push_constraint_vertex_indices: int32")
	.SetShapeFn( [] (::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->MakeShape({c->kUnknownDim}));
		c->set_output(1, c->MakeShape({c->kUnknownDim}));
		c->set_output(2, c->MakeShape({c->kUnknownDim, 2}));
		c->set_output(3, c->MakeShape({c->kUnknownDim, 2, 2}));
		return Status::OK();
	} );

class PushOffsetsOp : public OpKernel
{
public:

	explicit PushOffsetsOp(OpKernelConstruction* context) : OpKernel(context)
	{
	}

	void Compute(OpKernelContext* context) override
	{
		Tensor const &rotated_vertices_tensor = context->input(0);
		OP_REQUIRES(
			context,
			rotated_vertices_tensor.shape().dims() == 2 && rotated_vertices_tensor.shape().dim_size(1) == 3,
			errors::InvalidArgument("PushOffsets expects rotated_vertices to be 2D, with final dimension of size three")
		);
		auto const rotated_vertices = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const>(
			rotated_vertices_tensor.matrix<float>().data(),
			rotated_vertices_tensor.shape().dim_size(0),
			rotated_vertices_tensor.shape().dim_size(1)
		);
		auto const vertex_count = rotated_vertices.rows();

		Tensor const &faces_tensor = context->input(1);
		OP_REQUIRES(
			context,
			faces_tensor.shape().dims() == 2 && faces_tensor.shape().dim_size(1) == 3,
			errors::InvalidArgument("PushOffsets expects faces to be 2D, with final dimension of size three")
		);
		auto const faces =  Eigen::Map<Eigen::Matrix<int32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const>(
			faces_tensor.matrix<int32>().data(),
			faces_tensor.shape().dim_size(0),
			faces_tensor.shape().dim_size(1)
		);

		Tensor const &initial_offset_magnitudes_tensor = context->input(2);
		OP_REQUIRES(
			context,
			initial_offset_magnitudes_tensor.shape().dims() == 1 && initial_offset_magnitudes_tensor.shape().dim_size(0) == rotated_vertices_tensor.shape().dim_size(0),
			errors::InvalidArgument("PushOffsets expects initial_offset_magnitudes_tensor to be 1D, with size equal to first dimension of rotated_vertices")
		);
		auto const initial_offset_magnitudes = Eigen::Map<Eigen::ArrayXf const>(
			initial_offset_magnitudes_tensor.flat<float>().data(),
			initial_offset_magnitudes_tensor.shape().dim_size(0)
		);

		try {
			auto const result = get_pushed_offset_magnitudes_and_active_constraints(rotated_vertices, faces, initial_offset_magnitudes);

			Tensor *final_offset_magnitudes_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{vertex_count}, &final_offset_magnitudes_tensor));
			auto final_offset_magnitudes_tensor_data = final_offset_magnitudes_tensor->flat<float>();
			std::copy(
				result.final_offset_magnitudes.data(),
				result.final_offset_magnitudes.data() + result.final_offset_magnitudes.size(),
				final_offset_magnitudes_tensor_data.data()
			);

			Tensor *active_bound_constraint_vertex_indices_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{result.active_bound_constraint_vertex_indices.size()}, &active_bound_constraint_vertex_indices_tensor));
			auto active_bound_constraint_vertex_indices_tensor_data = active_bound_constraint_vertex_indices_tensor->flat<int32>();
			std::copy(
				result.active_bound_constraint_vertex_indices.data(),
				result.active_bound_constraint_vertex_indices.data() + result.active_bound_constraint_vertex_indices.size(),
				active_bound_constraint_vertex_indices_tensor_data.data()
			);

			assert(result.active_push_constraint_face_indices.size() == result.active_push_constraint_vertex_indices.size());
			auto const active_push_constraint_count = result.active_push_constraint_face_indices.size();
			
			Tensor *active_push_constraint_face_indices_tensor = nullptr, *active_push_constraint_vertex_indices_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{active_push_constraint_count, 2}, &active_push_constraint_face_indices_tensor));
			OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{active_push_constraint_count, 2, 2}, &active_push_constraint_vertex_indices_tensor));
			auto active_push_constraint_face_indices_tensor_data = active_push_constraint_face_indices_tensor->matrix<int32>();
			auto active_push_constraint_vertex_indices_tensor_data = active_push_constraint_vertex_indices_tensor->tensor<int32, 3>();

			for (int constraint_index = 0; constraint_index < active_push_constraint_count; ++constraint_index) {
				active_push_constraint_face_indices_tensor_data(constraint_index, 0) = result.active_push_constraint_face_indices[constraint_index].first;
				active_push_constraint_face_indices_tensor_data(constraint_index, 1) = result.active_push_constraint_face_indices[constraint_index].second;
				active_push_constraint_vertex_indices_tensor_data(constraint_index, 0, 0) = result.active_push_constraint_vertex_indices[constraint_index].first.first;
				active_push_constraint_vertex_indices_tensor_data(constraint_index, 0, 1) = result.active_push_constraint_vertex_indices[constraint_index].first.second;
				active_push_constraint_vertex_indices_tensor_data(constraint_index, 1, 0) = result.active_push_constraint_vertex_indices[constraint_index].second.first;
				active_push_constraint_vertex_indices_tensor_data(constraint_index, 1, 1) = result.active_push_constraint_vertex_indices[constraint_index].second.second;
			}
		} catch (GRBException const &e) {
			std::ostringstream error_ss;
			error_ss <<  "exception in Gurobi: " << e.getErrorCode() << ": " << e.getMessage();
			OP_REQUIRES(context, false, errors::InvalidArgument(error_ss.str()));
		} catch (std::exception const &e) {
			OP_REQUIRES(context, false, errors::InvalidArgument(e.what()));
		}
	}
};

REGISTER_KERNEL_BUILDER(Name("PushOffsets").Device(DEVICE_CPU), PushOffsetsOp);

