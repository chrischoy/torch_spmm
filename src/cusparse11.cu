#include <cusparse.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

Tensor _bmm_out_sparse_cuda(torch::Tensor const &rows,
                            torch::Tensor const &cols,
                            torch::Tensor const &vals, int64_t const dim_i,
                            int64_t const dim_j, torch::Tensor const &mat2,
                            bool deterministic) {
#if defined __HIP_PLATFORM_HCC__
  TORCH_CHECK(false, "bmm sparse-dense is not supported on HIP");
#elif defined(_WIN32) || defined(_WIN64)
  TORCH_CHECK(false, "bmm sparse-dense CUDA is not supported on Windows");
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 10010)

  TORCH_CHECK(!mat2.is_sparse(), "bmm_sparse: Tensor 'mat2' must be dense");
  TORCH_CHECK(mat2.dim() == 3,
              "bmm_sparse: Tensor 'mat2' must have 3 dims, but has ",
              mat2.dim());
  TORCH_CHECK(self.size(0) == mat2.size(0),
              "bmm_sparse: 'self.size(0)' and 'mat2.size(0)' must match");

  // int64_t dim_i = self.size(0);
  // int64_t dim_j = self.size(1);
  int64_t dim_k = mat2.size(1);

  torch::Tensor result = at::zeros({dim_k, dim_i}, mat2.options());

  if ((dim_j == 0) || (dim_k == 0)) {
    return result;
  }

  // Dense matrices have to be contiguous for cusparseSpMM to work
  const Tensor mat2_contig = mat2.contiguous();
  auto cusparse_handle = at::cuda::getCurrentCUDASparseHandle();

  torch::Scalar beta = 0;
  torch::Scalar alpha = 1;

  int64_t mat_el_begin_idx = 0;
  size_t workspace_buffer_size = 0;
  void *workspace_buffer = nullptr;

  deterministic = deterministic || at::globalContext().deterministic();
  cusparseSpMMAlg_t mm_alg =
      deterministic ? CUSPARSE_COOMM_ALG2 : CUSPARSE_COOMM_ALG1;

  // Iterate through each set of 2D matrices within the 3D
  // tensor inputs, performing a matrix multiply with each
  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "bmm_sparse_cuda", [&] {
    scalar_t alpha_val = alpha.to<scalar_t>();
    scalar_t beta_val = beta.to<scalar_t>();

    // Create tensors to view just the current set of matrices
    int64_t sparse_nnz = rows.numel();

    cudaDataType cuda_data_type = getTensorCudaDataType(mat2_contig);
    uint32_t *row_indices_ptr = reinterpret_cast<uint32_t *>(rows.data_ptr());
    uint32_t *col_indices_ptr = reinterpret_cast<uint32_t *>(cols.data_ptr());
    scalar_t *values_ptr = reinterpret_cast<scalar_t *>(values.data_ptr());
    scalar_t *mat2_ptr = reinterpret_cast<scalar_t *>(mat2_contig.data_ptr());
    scalar_t *result_ptr = reinterpret_cast<scalar_t *>(result.data_ptr());

    cusparseSpMatDescr_t sparse_descr;
    TORCH_CUDASPARSE_CHECK(cusparseCreateCoo(
        &sparse_descr,            //
        dim_i, dim_j, sparse_nnz, //
        reinterpret_cast<void *>(row_indices_ptr),
        reinterpret_cast<void *>(col_indices_ptr),
        reinterpret_cast<void *>(values_ptr), //
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cuda_data_type));

    cusparseDnMatDescr_t dense_descr;
    TORCH_CUDASPARSE_CHECK(
        cusparseCreateDnMat(&dense_descr,                       //
                            dim_k, dim_j, dim_k,                //
                            reinterpret_cast<void *>(mat2_ptr), //
                            cuda_data_type, CUSPARSE_ORDER_COL));

    cusparseDnMatDescr_t result_descr;
    TORCH_CUDASPARSE_CHECK(
        cusparseCreateDnMat(&result_descr,                        //
                            dim_i, dim_k, dim_i,                  //
                            reinterpret_cast<void *>(result_ptr), //
                            cuda_data_type, CUSPARSE_ORDER_COL));

    size_t required_workspace_buffer_size = 0;
    TORCH_CUDASPARSE_CHECK(cusparseSpMM_bufferSize(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE, (void *)&alpha_val, sparse_descr,
        dense_descr, (void *)&beta_val, result_descr, cuda_data_type, mm_alg,
        &required_workspace_buffer_size));

    if (required_workspace_buffer_size > workspace_buffer_size) {
      if (workspace_buffer != nullptr) {
        cudaFree(workspace_buffer);
      }
      workspace_buffer_size = required_workspace_buffer_size;
      cudaMallocManaged(&workspace_buffer, workspace_buffer_size);
    }
    TORCH_CUDASPARSE_CHECK(
        cusparseSpMM(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     CUSPARSE_OPERATION_TRANSPOSE, (void *)&alpha_val,
                     sparse_descr, dense_descr, (void *)&beta_val, result_descr,
                     cuda_data_type, mm_alg, workspace_buffer));
    TORCH_CUDASPARSE_CHECK(cusparseDestroySpMat(sparse_descr));
    TORCH_CUDASPARSE_CHECK(cusparseDestroyDnMat(dense_descr));
    TORCH_CUDASPARSE_CHECK(cusparseDestroyDnMat(result_descr));
  });

  // Need to transpose the result matrices since cusparse stores
  // them in column-major order in memory
  result.transpose_(0, 1);

  if (workspace_buffer != nullptr) {
    cudaFree(workspace_buffer);
  }

  return result;
#else
  TORCH_CHECK(false, "bmm sparse-dense requires CUDA 10.1 or greater");
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("coordinate_map_key_return", &minkowski::coordinate_map_key_return,
        "Minkowski Engine coordinate map key return test");
}
