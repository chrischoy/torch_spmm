#include "helper.cuh"

#include <cusparse.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <torch/extension.h>

template <typename th_int_type>
torch::Tensor coo_spmm(torch::Tensor const &rows, torch::Tensor const &cols,
                       torch::Tensor const &vals, int64_t const dim_i,
                       int64_t const dim_j, torch::Tensor const &mat2,
                       int64_t spmm_algorithm_id) {
#if defined __HIP_PLATFORM_HCC__
  TORCH_CHECK(false, "spmm sparse-dense is not supported on HIP");
#elif defined(_WIN32) || defined(_WIN64)
  TORCH_CHECK(false, "spmm sparse-dense CUDA is not supported on Windows");
#elif !defined(CUDART_VERSION)
  TORCH_CHECK(false, "CUDART_VERSION not defined");
#endif

  constexpr bool is_int32 = std::is_same<th_int_type, int32_t>::value;
  constexpr bool is_int64 = std::is_same<th_int_type, int64_t>::value;

  cusparseSpMMAlg_t mm_alg;
#if defined(CUDART_VERSION) && (CUDART_VERSION < 10010)
  TORCH_CHECK(false, "spmm sparse-dense requires CUDA 10.1 or greater");
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 10010) &&                  \
    (CUDART_VERSION < 11000)
  switch (spmm_algorithm_id) {
  case 1:
    mm_alg = CUSPARSE_COOMM_ALG1;
    break;
  case 2:
    mm_alg = CUSPARSE_COOMM_ALG2;
    break;
  case 3:
    mm_alg = CUSPARSE_COOMM_ALG3;
    break;
  default:
    TORCH_CHECK(false, "Invalid algorithm id.", spmm_algorithm_id);
    mm_alg = CUSPARSE_MM_ALG_DEFAULT;
  }
  TORCH_CHECK(is_int32, "int64 cusparseSpMM requires CUDA 11.0 or greater");
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 11000)
  switch (spmm_algorithm_id) {
  case 1:
    mm_alg = CUSPARSE_SPMM_COO_ALG1;
    break;
  case 2:
    mm_alg = CUSPARSE_SPMM_COO_ALG2;
    break;
  case 3:
    mm_alg = CUSPARSE_SPMM_COO_ALG3;
    break;
  case 4:
    // CUSPARSE_SPMM_CSR_ALG4 should be used with row-major layout, while
    // CUSPARSE_SPMM_CSR_ALG1, CUSPARSE_SPMM_CSR_ALG2, and
    // CUSPARSE_SPMM_CSR_ALG3 with column-major layout.
    mm_alg = CUSPARSE_SPMM_COO_ALG4;
    break;
  default:
    TORCH_CHECK(false, "Invalid algorithm id.", spmm_algorithm_id);
    mm_alg = CUSPARSE_SPMM_ALG_DEFAULT;
  }
  TORCH_CHECK(is_int32 || (is_int64 && (mm_alg == CUSPARSE_SPMM_COO_ALG4)));
#endif

  at::ScalarType int_scalar_type =
      is_int32 ? at::ScalarType::Int : at::ScalarType::Long;

  TORCH_CHECK(rows.scalar_type() == int_scalar_type, "int type mismatch.");

  TORCH_CHECK(rows.scalar_type() == cols.scalar_type(),
              "rows and cols must have the same scalar type.");
  TORCH_CHECK(rows.scalar_type() == cols.scalar_type(),
              "rows and cols must have the same scalar type.");
  TORCH_CHECK(vals.scalar_type() == mat2.scalar_type(),
              "vals and mat2 must have the same scalar type.");

  TORCH_CHECK(rows.is_cuda(), "rows must be CUDA, but got CPU");
  TORCH_CHECK(cols.is_cuda(), "cols must be CUDA, but got CPU");
  TORCH_CHECK(vals.is_cuda(), "vals must be CUDA, but got CPU");
  TORCH_CHECK(mat2.is_cuda(), "mat2 must be CUDA, but got CPU");
  TORCH_CHECK(at::cuda::check_device({rows, cols, vals, mat2}));

  TORCH_CHECK(mat2.dim() == 2, "Tensor 'mat2' must have 2 dims, but has ",
              mat2.dim());

  // int64_t dim_i = self.size(0);
  // int64_t dim_j = self.size(1);
  int64_t dim_k = mat2.size(1);

  torch::Tensor result = at::empty({dim_k, dim_i}, mat2.options());

  if ((dim_j == 0) || (dim_k == 0)) {
    return result;
  }

  // Dense matrices have to be contiguous for cusparseSpMM to work
  torch::Tensor const mat2_contig = mat2.contiguous();
  auto cusparse_handle = at::cuda::getCurrentCUDASparseHandle();

  torch::Scalar beta = 0;
  torch::Scalar alpha = 1;

  size_t workspace_buffer_size = 0;
  void *workspace_buffer = nullptr;

  cusparseSpMatDescr_t sparse_descr;
  cusparseDnMatDescr_t dense_descr;
  cusparseDnMatDescr_t result_descr;

  // Iterate through each set of 2D matrices within the 3D
  // tensor inputs, performing a matrix multiply with each
  AT_DISPATCH_FLOATING_TYPES(vals.scalar_type(), "coo_spmm", [&] {
    scalar_t alpha_val = alpha.to<scalar_t>();
    scalar_t beta_val = beta.to<scalar_t>();

    // Create tensors to view just the current set of matrices
    int64_t sparse_nnz = rows.numel();

    cudaDataType cuda_data_type = getTensorCudaDataType(mat2_contig);
    th_int_type *row_indices_ptr =
        reinterpret_cast<th_int_type *>(rows.data_ptr());
    th_int_type *col_indices_ptr =
        reinterpret_cast<th_int_type *>(cols.data_ptr());
    scalar_t *values_ptr = reinterpret_cast<scalar_t *>(vals.data_ptr());
    scalar_t *mat2_ptr = reinterpret_cast<scalar_t *>(mat2_contig.data_ptr());
    scalar_t *result_ptr = reinterpret_cast<scalar_t *>(result.data_ptr());

    if (mm_alg == CUSPARSE_SPMM_COO_ALG4) {
      TORCH_CUDASPARSE_CHECK(cusparseCreateCoo(
          &sparse_descr,            //
          dim_i, dim_j, sparse_nnz, //
          reinterpret_cast<void *>(row_indices_ptr),
          reinterpret_cast<void *>(col_indices_ptr),
          reinterpret_cast<void *>(values_ptr), //
          std::is_same<th_int_type, int32_t>::value ? CUSPARSE_INDEX_32I
                                                    : CUSPARSE_INDEX_64I,
          CUSPARSE_INDEX_BASE_ZERO, cuda_data_type));

      TORCH_CUDASPARSE_CHECK(
          cusparseCreateDnMat(&dense_descr,                       //
                              dim_j, dim_k, dim_j,                //
                              reinterpret_cast<void *>(mat2_ptr), //
                              cuda_data_type, CUSPARSE_ORDER_ROW));

      TORCH_CUDASPARSE_CHECK(
          cusparseCreateDnMat(&result_descr,                        //
                              dim_i, dim_k, dim_i,                  //
                              reinterpret_cast<void *>(result_ptr), //
                              cuda_data_type, CUSPARSE_ORDER_ROW));

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
      TORCH_CUDASPARSE_CHECK(cusparseSpMM(cusparse_handle,                  //
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, //
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, //
                                          (void *)&alpha_val,               //
                                          sparse_descr, dense_descr,        //
                                          (void *)&beta_val, result_descr,  //
                                          cuda_data_type, mm_alg,
                                          workspace_buffer));
    } else {
      TORCH_CUDASPARSE_CHECK(cusparseCreateCoo(
          &sparse_descr,            //
          dim_i, dim_j, sparse_nnz, //
          reinterpret_cast<void *>(row_indices_ptr),
          reinterpret_cast<void *>(col_indices_ptr),
          reinterpret_cast<void *>(values_ptr), //
          std::is_same<th_int_type, int32_t>::value ? CUSPARSE_INDEX_32I
                                                    : CUSPARSE_INDEX_64I,
          CUSPARSE_INDEX_BASE_ZERO, cuda_data_type));

      TORCH_CUDASPARSE_CHECK(
          cusparseCreateDnMat(&dense_descr,                       //
                              dim_k, dim_j, dim_k,                //
                              reinterpret_cast<void *>(mat2_ptr), //
                              cuda_data_type, CUSPARSE_ORDER_COL));

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
      TORCH_CUDASPARSE_CHECK(cusparseSpMM(cusparse_handle,                  //
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, //
                                          CUSPARSE_OPERATION_TRANSPOSE,     //
                                          (void *)&alpha_val,               //
                                          sparse_descr, dense_descr,        //
                                          (void *)&beta_val, result_descr,  //
                                          cuda_data_type, mm_alg,
                                          workspace_buffer));
      // Need to transpose the result matrices since cusparse stores
      // them in column-major order in memory
      result.transpose_(0, 1);
    }
  });

  TORCH_CUDASPARSE_CHECK(cusparseDestroySpMat(sparse_descr));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyDnMat(dense_descr));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyDnMat(result_descr));

  if (workspace_buffer != nullptr) {
    cudaFree(workspace_buffer);
  }

  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("coo_spmm_int32", &coo_spmm<int32_t>, "sparse matrix x dense matrix");
  m.def("coo_spmm_int64", &coo_spmm<int64_t>, "sparse matrix x dense matrix");
}
