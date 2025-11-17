#include <cuda_runtime.h>

#include "cuda/dummy_kernel.cuh"

__global__ void AddOneKernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] += 1.0f;
}

extern "C" void DummyCudaAddOne(float* host_data, int n) {
  float* dev = nullptr;
  size_t bytes = sizeof(float) * n;

  cudaError_t err;
  err = cudaMalloc(&dev, bytes);
  if (err != cudaSuccess) return;

  cudaMemcpy(dev, host_data, bytes, cudaMemcpyHostToDevice);

  int block = 256;
  int grid = (n + block - 1) / block;
  AddOneKernel<<<grid, block>>>(dev, n);
  cudaDeviceSynchronize();

  cudaMemcpy(host_data, dev, bytes, cudaMemcpyDeviceToHost);
  cudaFree(dev);
}