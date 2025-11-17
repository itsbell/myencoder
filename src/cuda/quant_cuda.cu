// quant_cuda.cu
#include <cuda_runtime.h>

// VP8Matrix 정의가 들어 있는 헤더를 include 해야 함.
#include "src/dsp/dsp.h"

#include "src/enc/vp8i_enc.h" // QUANTDIV 함수

// enc.c에 static으로 있는 것과 동일한 내용이지만, 디바이스 전용 상수로 선언
__device__ __constant__ uint8_t kZigzagDev[16] = {0, 1,  4,  8,  5, 2,  3,  6,
                                                  9, 12, 13, 10, 7, 11, 14, 15};


// QuantizeBlock_C에서 쓰이던 선언과 맞추기 위해
extern "C" int QuantizeBlock_CUDA(int16_t in[16], int16_t out[16],
                                  const VP8Matrix* WEBP_RESTRICT const mtx);

// 예시 코드: 디바이스 커널, threadIdx.x == 0 만 루프 실행
__global__ void QuantizeBlockKernel(int16_t* in, int16_t* out,
                                    const VP8Matrix* mtx, int* d_last_nonzero) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  int last = -1;
  for (int n = 0; n < 16; ++n) {
    const int j = kZigzagDev[n];
    const int sign = (in[j] < 0);
    const uint32_t coeff = (sign ? -in[j] : in[j]) + mtx->sharpen[j];
    if (coeff > mtx->zthresh[j]) {
      const uint32_t Q = mtx->q[j];
      const uint32_t iQ = mtx->iq[j];
      const uint32_t B = mtx->bias[j];
      // int level = QUANTDIV(coeff, iQ, B);
      int level = (int)((coeff * iQ + B) >> QFIX);
      if (level > MAX_LEVEL) level = MAX_LEVEL;
      if (sign) level = -level;
      in[j] = level * (int)Q;
      out[n] = level;
      if (level) last = n;
    } else {
      out[n] = 0;
      in[j] = 0;
    }
  }
  *d_last_nonzero = last;
}

// cudaMalloc / cudaFree를 한 번만 하고 재사용하기
static int16_t* d_in = nullptr;
static int16_t* d_out = nullptr;
static VP8Matrix* d_mtx = nullptr;
static int* d_last = nullptr;
static bool s_inited = false;
static const VP8Matrix* s_last_mtx = nullptr;

// 예시 코드: 한 블록만 GPU에 보내는 단순 래퍼
extern "C" int QuantizeBlock_CUDA(int16_t in[16], int16_t out[16],
                                  const VP8Matrix* WEBP_RESTRICT const mtx) {
  if (!s_inited) {
    cudaMalloc(&d_in, sizeof(int16_t) * 16);
    cudaMalloc(&d_out, sizeof(int16_t) * 16);
    cudaMalloc(&d_mtx, sizeof(VP8Matrix));
    cudaMalloc(&d_last, sizeof(int));
    s_inited = true;
  }

  int last = -1;

  cudaMemcpy(d_in, in, sizeof(int16_t) * 16, cudaMemcpyHostToDevice);

  if (mtx != s_last_mtx) {
    cudaMemcpy(d_mtx, mtx, sizeof(VP8Matrix), cudaMemcpyHostToDevice);
    s_last_mtx = mtx;
  }

  QuantizeBlockKernel<<<1, 1>>>(d_in, d_out, d_mtx, d_last);
  cudaDeviceSynchronize();

  cudaMemcpy(in, d_in, sizeof(int16_t) * 16, cudaMemcpyDeviceToHost);
  cudaMemcpy(out, d_out, sizeof(int16_t) * 16, cudaMemcpyDeviceToHost);
  cudaMemcpy(&last, d_last, sizeof(int), cudaMemcpyDeviceToHost);

  return (last >= 0);
}
