#pragma once

#include <cusparse.h>

#include <torch/extension.h>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")

void Xcoosort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz,
                            const int *cooRows, const int *cooCols,
                            size_t *pBufferSizeInBytes) {
  TORCH_CHECK(
      (m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
      "Xcoosort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
      INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  TORCH_CUDASPARSE_CHECK(cusparseXcoosort_bufferSizeExt(
      handle, i_m, i_n, i_nnz, cooRows, cooCols, pBufferSizeInBytes));
}

void XcoosortByRow(int64_t m, int64_t n, int64_t nnz, int *cooRows,
                   int *cooCols, int *P, void *pBuffer) {
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
              "XcoosortByRow only supports m, n, nnz with the bound [val] <= ",
              INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  TORCH_CUDASPARSE_CHECK(cusparseXcoosortByRow(handle, i_m, i_n, i_nnz, cooRows,
                                               cooCols, P, pBuffer));
}

cusparseOperation_t convertTransToCusparseOperation(char trans) {
  if (trans == 't')
    return CUSPARSE_OPERATION_TRANSPOSE;
  else if (trans == 'n')
    return CUSPARSE_OPERATION_NON_TRANSPOSE;
  else if (trans == 'c')
    return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else {
    AT_ERROR("trans must be one of: t, n, c");
  }
}

cudaDataType getTensorCudaDataType(torch::Tensor const &self) {
  cudaDataType cuda_data_type;
  switch (self.scalar_type()) {
  case ScalarType::Float:
    cuda_data_type = CUDA_R_32F;
    break;
  case ScalarType::Double:
    cuda_data_type = CUDA_R_64F;
    break;
  default:
    TORCH_CHECK(false, "Tensor types must be either float32 or float64");
    break;
  }
  return cuda_data_type;
}
