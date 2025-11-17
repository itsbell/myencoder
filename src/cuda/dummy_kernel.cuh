// src/cuda/dummy_kernel.cuh  (¿¹½Ã)
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void DummyCudaAddOne(float* host_data, int n);

#ifdef __cplusplus
}
#endif
