// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NCNN_MAT_PIXEL_H
#define NCNN_MAT_PIXEL_H

#if !defined(NCNN_EXPORT)
#define NCNN_EXPORT
#endif

namespace ncnn {

// misc function
// #if NCNN_PIXEL
// convert yuv420sp(nv21) to rgb, the fast approximate version
NCNN_EXPORT void yuv420sp2rgb(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
// convert yuv420sp(nv12) to rgb, the fast approximate version
NCNN_EXPORT void yuv420sp2rgb_nv12(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
// convert yuv420sp(nv21) to rgb with half resize, the faster approximate version
NCNN_EXPORT void yuv420sp2rgb_half(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
// image pixel bilinear resize
NCNN_EXPORT void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
NCNN_EXPORT void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
NCNN_EXPORT void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
NCNN_EXPORT void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
// image pixel bilinear resize with stride(bytes-per-row) parameter
NCNN_EXPORT void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
NCNN_EXPORT void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
NCNN_EXPORT void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
NCNN_EXPORT void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
// image pixel bilinear resize, convenient wrapper for yuv420sp(nv21/nv12)
NCNN_EXPORT void resize_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
// #endif // NCNN_PIXEL

// #if NCNN_PIXEL_DRAWING
// draw rectangle, set thickness -1 for filled rectangle, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_rectangle_c1(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
NCNN_EXPORT void draw_rectangle_c2(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
NCNN_EXPORT void draw_rectangle_c3(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
NCNN_EXPORT void draw_rectangle_c4(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
// draw rectangle with stride(bytes-per-row) parameter, set thickness -1 for filled rectangle, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_rectangle_c1(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
NCNN_EXPORT void draw_rectangle_c2(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
NCNN_EXPORT void draw_rectangle_c3(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
NCNN_EXPORT void draw_rectangle_c4(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
// draw rectangle, convenient wrapper for yuv420sp(nv21/nv12), set thickness -1 for filled rectangle, the color YUV_ is little-endian encoded
NCNN_EXPORT void draw_rectangle_yuv420sp(unsigned char* yuv420sp, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
// draw circle, set thickness -1 for filled circle, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_circle_c1(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
NCNN_EXPORT void draw_circle_c2(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
NCNN_EXPORT void draw_circle_c3(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
NCNN_EXPORT void draw_circle_c4(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
// draw circle with stride(bytes-per-row) parameter, set thickness -1 for filled circle, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_circle_c1(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
NCNN_EXPORT void draw_circle_c2(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
NCNN_EXPORT void draw_circle_c3(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
NCNN_EXPORT void draw_circle_c4(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
// draw circle, convenient wrapper for yuv420sp(nv21/nv12), set thickness -1 for filled circle, the color YUV_ is little-endian encoded
NCNN_EXPORT void draw_circle_yuv420sp(unsigned char* yuv420sp, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
// draw line, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_line_c1(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
NCNN_EXPORT void draw_line_c2(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
NCNN_EXPORT void draw_line_c3(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
NCNN_EXPORT void draw_line_c4(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
// draw line with stride(bytes-per-row) parameter, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_line_c1(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
NCNN_EXPORT void draw_line_c2(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
NCNN_EXPORT void draw_line_c3(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
NCNN_EXPORT void draw_line_c4(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
// draw line, convenient wrapper for yuv420sp(nv21/nv12), the color YUV_ is little-endian encoded
NCNN_EXPORT void draw_line_yuv420sp(unsigned char* yuv420sp, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
// resolve text bounding box size
NCNN_EXPORT void get_text_drawing_size(const char* text, int fontpixelsize, int* w, int* h);
// draw ascii printables and newline, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_text_c1(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
NCNN_EXPORT void draw_text_c2(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
NCNN_EXPORT void draw_text_c3(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
NCNN_EXPORT void draw_text_c4(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
// draw ascii printables and newline with stride(bytes-per-row) parameter, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_text_c1(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
NCNN_EXPORT void draw_text_c2(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
NCNN_EXPORT void draw_text_c3(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
NCNN_EXPORT void draw_text_c4(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
// draw ascii printables and newline, convenient wrapper for yuv420sp(nv21/nv12), the color YUV_ is little-endian encoded
NCNN_EXPORT void draw_text_yuv420sp(unsigned char* yuv420sp, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
// #endif // NCNN_PIXEL_DRAWING

} // namespace ncnn

#endif  // NCNN_MAT_PIXEL_H
