// Intra16.cu
#include <cuda_runtime.h>
extern "C" {
#include "src/cuda/Intra16.cuh"
#include "src/dsp/dsp.h"       // VP8FTransform2, VP8ITransform 등
#include "src/enc/vp8i_enc.h"  // VP8Encoder, VP8Matrix, VP8Scan 등
}

// Coefficient type.
enum {
  TYPE_I16_AC = 0,
  TYPE_I16_DC = 1,
  TYPE_CHROMA_A = 2,
  TYPE_I4_AC = 3
};  // quant_enc.c 와 동일하게 선언

extern "C" {
int TrellisQuantizeBlock(const VP8Encoder* enc, int16_t in[16], int16_t out[16],
                         int ctx0, int coeff_type, const VP8Matrix* mtx,
                         int lambda);  // quant_enc.c 정의
}

extern "C" {
int ReconstructIntra16_CUDA(const VP8Intra16Job* job, int num_jobs) {
  (void)num_jobs;
  int nz = 0;
  int n;
  int16_t tmp[16][16], dc_tmp[16];

  // 1) FTransform2
  for (n = 0; n < 16; n += 2) {
    VP8FTransform2(job->src + VP8Scan[n], job->ref + VP8Scan[n], tmp[n]);
  }

  // 2) DC WHT + 양자화
  VP8FTransformWHT(tmp[0], dc_tmp);
  nz |= VP8EncQuantizeBlockWHT(dc_tmp, job->y_dc_levels, job->y2) << 24;

  // 3) AC 양자화
  if (job->do_trellis) {
    int x, y;
    int top_nz[4], left_nz[4];
    for (int i = 0; i < 4; ++i) {
      top_nz[i] = job->top_nz[i];
      left_nz[i] = job->left_nz[i];
    }
    for (y = 0, n = 0; y < 4; ++y) {
      for (x = 0; x < 4; ++x, ++n) {
        const int ctx = top_nz[x] + left_nz[y];
        int16_t* const ac = job->y_ac_levels + n * 16;
        const int non_zero =
            TrellisQuantizeBlock(job->enc, tmp[n], ac, ctx, TYPE_I16_AC,
                                 job->y1, job->lambda_trellis_i16);
        top_nz[x] = left_nz[y] = non_zero;
        ac[0] = 0;
        nz |= non_zero << n;
      }
    }
  } else {
    for (n = 0; n < 16; n += 2) {
      tmp[n][0] = tmp[n + 1][0] = 0;
      nz |= VP8EncQuantize2Blocks(tmp[n], job->y_ac_levels + n * 16, job->y1)
            << n;
    }
  }

  // 4) 역 WHT + 역변환
  VP8TransformWHT(dc_tmp, tmp[0]);
  for (n = 0; n < 16; n += 2) {
    VP8ITransform(job->ref + VP8Scan[n], tmp[n], job->dst + VP8Scan[n], 1);
  }

  return nz;
}
}