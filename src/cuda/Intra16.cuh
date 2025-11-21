// Intra16.cuh
#ifndef INTRA16_CUH_
#define INTRA16_CUH_

#include <stdint.h>

#include "src/enc/vp8i_enc.h"  // VP8Matrix, VP8SegmentInfo, VP8Encoder 등 선언
// ↑ 정확한 헤더 이름은 프로젝트에서 실제로 사용하는 걸 확인해서 맞춰줘.
//   (내가 그 헤더 파일 내용은 지금 못 보고 있어서, 파일명은 추측이야. 이 부분은
//   꼭 직접 확인!)

#ifdef __cplusplus
extern "C" {
#endif

// MB 하나에 대한 작업을 정의하는 job 구조체 (GPU-friendly하게 단순
// 포인터/스칼라만)
typedef struct {
  const VP8Encoder* enc;
  // 입력 Y 포인터들
  const uint8_t* src;  // it->yuv_in + Y_OFF_ENC
  const uint8_t* ref;  // it->yuv_p   + VP8I16ModeOffsets[mode]
  int src_stride;
  int ref_stride;

  // 양자화 매트릭스
  const VP8Matrix* y1;  // dqm->y1
  const VP8Matrix* y2;  // dqm->y2

  // 트렐리스 정보
  int do_trellis;  // it->do_trellis 값 복사
  int top_nz[4];  // I16에서 실제로 쓰는 건 [0..3]
  int left_nz[4];
  int lambda_trellis_i16;

  // 출력 버퍼
  int16_t* y_dc_levels;  // rd->y_dc_levels
  int16_t* y_ac_levels;  // &rd->y_ac_levels[0][0] 처럼 편평화해서 넘기는 걸 추천
  uint8_t* dst;   // yuv_out

  // 결과 nz를 GPU에서 계산해서 돌려주고 싶으면 여기에 별도 필드를 추가해도 됨
} VP8Intra16Job;

int ReconstructIntra16_CUDA(const VP8Intra16Job* jobs, int num_jobs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // INTRA16_CUH_
