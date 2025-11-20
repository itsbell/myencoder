// yuv.cuh
#ifndef YUV_CUH_
#define YUV_CUH_

#ifdef __cplusplus
extern "C" {
#endif

void WebPImportYUVAFromRGBA_CUDA(const uint8_t* r_ptr, const uint8_t* g_ptr,
                                 const uint8_t* b_ptr, const uint8_t* a_ptr,
                                 int step,        // bytes per pixel
                                 int rgb_stride,  // bytes per scanline
                                 int has_alpha, int width, int height,
                                 uint16_t* tmp_rgb, int y_stride, int uv_stride,
                                 int a_stride, uint8_t* dst_y, uint8_t* dst_u,
                                 uint8_t* dst_v, uint8_t* dst_a);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // YUV_CUH_