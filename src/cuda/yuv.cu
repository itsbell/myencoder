// yuv.cu
#include <cuda_runtime.h>

extern "C" {
    #include "src/dsp/yuv.h"
    #include "src/cuda/yuv.cuh"
}

// 고정소수점 계수는 yuv.h 의 VP8RGBToY/U/V 와 동일하게 맞춤.
__device__ inline uint8_t DevRGBToY(int r, int g, int b) {
  const int luma = 16839 * r + 33059 * g + 6420 * b;
  const int y = (luma + YUV_HALF + (16 << YUV_FIX)) >> YUV_FIX;
  return (uint8_t)y;
}

__device__ inline int DevClipUV(int uv, int rounding) {
  uv = (uv + rounding + (128 << (YUV_FIX + 2))) >> (YUV_FIX + 2);
  if ((uv & ~0xff) == 0) return uv;
  return (uv < 0) ? 0 : 255;
}

__device__ inline uint8_t DevRGBToU(int r, int g, int b, int rounding) {
  const int u = -9719 * r - 19081 * g + 28800 * b;
  return (uint8_t)DevClipUV(u, rounding);
}

__device__ inline uint8_t DevRGBToV(int r, int g, int b, int rounding) {
  const int v = +28800 * r - 24116 * g - 4684 * b;
  return (uint8_t)DevClipUV(v, rounding);
}

__global__ void WebPImportYUVAFromRGBAKernel(
    const uint8_t* pR, const uint8_t* pG, const uint8_t* pB, int nStep,
    int nRGBStride, int nWidth, int nPairRows,  // pair_rows = height >> 1
    uint8_t* pY,                               // 크기: (2 * pair_rows) * width
    uint8_t* pU,                               // 크기: pair_rows * uv_width
    uint8_t* pV,                               // 크기: pair_rows * uv_width
    int nYStride,                          // 여기서는 width 로 사용
    int nUVStride)                         // 여기서는 uv_width 로 사용
{
  const int nUVX = blockIdx.x * blockDim.x + threadIdx.x;
  const int nUVY = blockIdx.y * blockDim.y + threadIdx.y;

  const int nUVWidth = (nWidth + 1) >> 1;

  if (nUVX >= nUVWidth || nUVY >= nPairRows)
      return;

  const int x0 = 2 * nUVX;
  const int y0 = 2 * nUVY;
  const int x1 = min(x0 + 1, nWidth - 1);
  const int y1 = y0 + 1;  // y_rows_gpu = 2 * pair_rows 이라서 항상 유효

  // 1. 입력 RGB 가져오기
  const uint8_t* pRow0_R = pR + y0 * nRGBStride;
  const uint8_t* pRow1_R = pR + y1 * nRGBStride;
  const uint8_t* pRow0_G = pG + y0 * nRGBStride;
  const uint8_t* pRow1_G = pG + y1 * nRGBStride;
  const uint8_t* pRow0_B = pB + y0 * nRGBStride;
  const uint8_t* pRow1_B = pB + y1 * nRGBStride;

  const int nIdx0 = x0 * nStep;
  const int nIdx1 = x1 * nStep;

  const int r00 = pRow0_R[nIdx0];
  const int g00 = pRow0_G[nIdx0];
  const int b00 = pRow0_B[nIdx0];

  const int r01 = pRow0_R[nIdx1];
  const int g01 = pRow0_G[nIdx1];
  const int b01 = pRow0_B[nIdx1];

  const int r10 = pRow1_R[nIdx0];
  const int g10 = pRow1_G[nIdx0];
  const int b10 = pRow1_B[nIdx0];

  const int r11 = pRow1_R[nIdx1];
  const int g11 = pRow1_G[nIdx1];
  const int b11 = pRow1_B[nIdx1];

  // 2. Y 채널 쓰기 (각 픽셀별)
  uint8_t* pRow0_Y = pY + y0 * nYStride;
  uint8_t* pRow1_Y = pY + y1 * nYStride;

  pRow0_Y[x0] = DevRGBToY(r00, g00, b00);
  if (x1 < nWidth)
      pRow0_Y[x1] = DevRGBToY(r01, g01, b01);

  pRow1_Y[x0] = DevRGBToY(r10, g10, b10);
  if (x1 < nWidth)
      pRow1_Y[x1] = DevRGBToY(r11, g11, b11);

  // 3. U/V 채널 (2x2 블록 평균)
  int nSumR, nSumG, nSumB;

  // width 가 홀수일 때 마지막 열은 CPU의 SUM2 / AccumulateRGB 동작과
  // 비슷하게 2개만 쓰도록 조정 (대략적인 맞추기)
  const bool bIsLastOddCol = ((nWidth & 1) && (x0 == nWidth - 1));

  if (!bIsLastOddCol)
  {
    nSumR = r00 + r01 + r10 + r11;
    nSumG = g00 + g01 + g10 + g11;
    nSumB = b00 + b01 + b10 + b11;
  }
  else
  {
    // 마지막 홀수 열: 위/아래 두 개만 사용하고 *2 해서 스케일 비슷하게 맞춤
    nSumR = 2 * (r00 + r10);
    nSumG = 2 * (g00 + g10);
    nSumB = 2 * (b00 + b10);
  }

  const int nRounding_uv = YUV_HALF << 2;
  const int nUVIndex = nUVY * nUVStride + nUVX;

  pU[nUVIndex] = DevRGBToU(nSumR, nSumG, nSumB, nRounding_uv);
  pV[nUVIndex] = DevRGBToV(nSumR, nSumG, nSumB, nRounding_uv);
}

extern "C" {
void WebPImportYUVAFromRGBA_CUDA(const uint8_t* r_ptr, const uint8_t* g_ptr,
                                 const uint8_t* b_ptr, const uint8_t* a_ptr,
                                 int nStep,        // bytes per pixel
                                 int nRGBStride,  // bytes per scanline
                                 int has_alpha, int nWidth, int nHeight,
                                 uint16_t* tmp_rgb, int y_stride, int uv_stride,
                                 int a_stride, uint8_t* dst_y, uint8_t* dst_u,
                                 uint8_t* dst_v, uint8_t* dst_a) {
  // height >> 1 쌍까지만 GPU가 처리하고, 마지막 홀수 줄은 picture_csp_enc.c에서
  // WebPImportYUVAFromRGBALastLine()이 처리
  const int nUVWidth = (nWidth + 1) >> 1;
  const int nUVHeight = nHeight >> 1;
  const int nRows = nUVHeight * 2;  // GPU가 처리할 Y 영역의 실제 줄 수

  const size_t nRSize = (size_t)nHeight * nRGBStride;
  const size_t nGSize = (size_t)nHeight * nRGBStride;
  const size_t nBSize = (size_t)nHeight * nRGBStride;

  const size_t nYSize = (size_t)nRows * nWidth;          // YUV의 Y
  const size_t nUVSize = (size_t)nUVHeight * nUVWidth;  // YUV의 UV

  uint8_t *pR = NULL, *pG = NULL, *pB = NULL;  // RGB (Kernel IN)
  uint8_t *pY = NULL, *pU = NULL, *pV = NULL;  // YUV (Kernel OUT)

  // 1. GPU 메모리 할당
  cudaMalloc(&pR, nRSize);
  cudaMalloc(&pG, nGSize);
  cudaMalloc(&pB, nBSize);

  cudaMalloc(&pY, nYSize);
  cudaMalloc(&pU, nUVSize);
  cudaMalloc(&pV, nUVSize);

  // 2. CPU -> GPU 메모리 복사
  cudaMemcpy(pR, r_ptr, nRSize, cudaMemcpyHostToDevice);
  cudaMemcpy(pG, g_ptr, nGSize, cudaMemcpyHostToDevice);
  cudaMemcpy(pB, b_ptr, nBSize, cudaMemcpyHostToDevice);

  // 3. 커널 런치
  const dim3 block(16, 16);
  const dim3 grid((nUVWidth + block.x - 1) / block.x, (nUVHeight + block.y - 1) / block.y);

  WebPImportYUVAFromRGBAKernel<<<grid, block>>>(pR, pG, pB, nStep, nRGBStride,
                                                nWidth, nUVHeight, pY, pU, pV,
                                                nWidth, nUVWidth);
  cudaDeviceSynchronize();

  // 4. GPU -> CPU 메모리 복사
  for (int y = 0; y < nRows; ++y)
  {
    cudaMemcpy(dst_y + (ptrdiff_t)y * y_stride, pY + (size_t)y * nWidth, (size_t)nWidth, cudaMemcpyDeviceToHost);
  }
  for (int row = 0; row < nUVHeight; ++row)
  {
    cudaMemcpy(dst_u + (ptrdiff_t)row * uv_stride, pU + (size_t)row * nUVWidth, (size_t)nUVWidth, cudaMemcpyDeviceToHost);
    cudaMemcpy(dst_v + (ptrdiff_t)row * uv_stride, pV + (size_t)row * nUVWidth, (size_t)nUVWidth, cudaMemcpyDeviceToHost);
  }

  // 5. GPU 메모리 할당 해제
  cudaFree(pR);
  cudaFree(pG);
  cudaFree(pB);

  cudaFree(pY);
  cudaFree(pU);
  cudaFree(pV);
}
}