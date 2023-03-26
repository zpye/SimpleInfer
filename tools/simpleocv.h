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

#ifndef NCNN_SIMPLEOCV_H
#define NCNN_SIMPLEOCV_H

// #include "platform.h"

// #if NCNN_SIMPLEOCV

// #include <limits.h>
// #include <string.h>
// #include "allocator.h"
// #include "mat.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <climits>
#include <cstring>
#include <cstdlib>

#include <algorithm>
#include <list>
#include <vector>
#include <string>

#if defined(_MSC_VER) || defined(__GNUC__)
#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max
#endif

// #ifndef NCNN_XADD
// using ncnn::NCNN_XADD;
// #endif

#define NCNN_EXPORT

#define NCNN_LOGE(...)

// from ncnn allocator.h

// the alignment of all the allocated buffers
#if NCNN_AVX512
#define NCNN_MALLOC_ALIGN 64
#elif NCNN_AVX
#define NCNN_MALLOC_ALIGN 32
#else
#define NCNN_MALLOC_ALIGN 16
#endif

// we have some optimized kernels that may overread buffer a bit in loop
// it is common to interleave next-loop data load with arithmetic instructions
// allocating more bytes keeps us safe from SEGV_ACCERR failure
#define NCNN_MALLOC_OVERREAD 64

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp>
static inline _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

static inline void* fastMalloc(size_t size)
{
#if _MSC_VER
    return _aligned_malloc(size, NCNN_MALLOC_ALIGN);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
    void* ptr = 0;
    if (posix_memalign(&ptr, NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD))
        ptr = 0;
    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD);
#else
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + NCNN_MALLOC_ALIGN + NCNN_MALLOC_OVERREAD);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, NCNN_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
#endif
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
#if _MSC_VER
        _aligned_free(ptr);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
        free(ptr);
#elif __ANDROID__ && __ANDROID_API__ < 17
        free(ptr);
#else
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
#endif
    }
}

static inline int NCNN_XADD(int* addr, int delta)
{
    int tmp = *addr;
    *addr += delta;
    return tmp;
}

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

enum
{
    CV_LOAD_IMAGE_UNCHANGED = -1,
    CV_LOAD_IMAGE_GRAYSCALE = 0,
    CV_LOAD_IMAGE_COLOR = 1,
};

enum
{
    CV_IMWRITE_JPEG_QUALITY = 1
};

// minimal opencv style data structure implementation
namespace cv {

template<typename _Tp>
static inline _Tp saturate_cast(int v)
{
    return _Tp(v);
}
template<>
inline uchar saturate_cast<uchar>(int v)
{
    return (uchar)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0);
}

template<typename _Tp>
struct Scalar_
{
    Scalar_()
    {
        v[0] = 0;
        v[1] = 0;
        v[2] = 0;
        v[3] = 0;
    }
    Scalar_(_Tp _v0)
    {
        v[0] = _v0;
        v[1] = 0;
        v[2] = 0;
        v[3] = 0;
    }
    Scalar_(_Tp _v0, _Tp _v1, _Tp _v2)
    {
        v[0] = _v0;
        v[1] = _v1;
        v[2] = _v2;
        v[3] = 0;
    }
    Scalar_(_Tp _v0, _Tp _v1, _Tp _v2, _Tp _v3)
    {
        v[0] = _v0;
        v[1] = _v1;
        v[2] = _v2;
        v[3] = _v3;
    }

    const _Tp operator[](const int i) const
    {
        return v[i];
    }

    _Tp operator[](const int i)
    {
        return v[i];
    }

    _Tp v[4];
};

typedef Scalar_<uchar> Scalar;

template<typename _Tp>
struct Point_
{
    Point_()
        : x(0), y(0)
    {
    }
    Point_(_Tp _x, _Tp _y)
        : x(_x), y(_y)
    {
    }

    template<typename _Tp2>
    operator Point_<_Tp2>() const
    {
        return Point_<_Tp2>(saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y));
    }

    _Tp x;
    _Tp y;
};

typedef Point_<int> Point;
typedef Point_<float> Point2f;

template<typename _Tp>
struct Size_
{
    Size_()
        : width(0), height(0)
    {
    }
    Size_(_Tp _w, _Tp _h)
        : width(_w), height(_h)
    {
    }

    template<typename _Tp2>
    operator Size_<_Tp2>() const
    {
        return Size_<_Tp2>(saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
    }

    _Tp width;
    _Tp height;
};

typedef Size_<int> Size;
typedef Size_<float> Size2f;

template<typename _Tp>
struct Rect_
{
    Rect_()
        : x(0), y(0), width(0), height(0)
    {
    }
    Rect_(_Tp _x, _Tp _y, _Tp _w, _Tp _h)
        : x(_x), y(_y), width(_w), height(_h)
    {
    }
    Rect_(Point_<_Tp> _p, Size_<_Tp> _size)
        : x(_p.x), y(_p.y), width(_size.width), height(_size.height)
    {
    }

    template<typename _Tp2>
    operator Rect_<_Tp2>() const
    {
        return Rect_<_Tp2>(saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y), saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
    }

    _Tp x;
    _Tp y;
    _Tp width;
    _Tp height;

    // area
    _Tp area() const
    {
        return width * height;
    }
};

template<typename _Tp>
static inline Rect_<_Tp>& operator&=(Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    _Tp x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    a.width = std::min(a.x + a.width, b.x + b.width) - x1;
    a.height = std::min(a.y + a.height, b.y + b.height) - y1;
    a.x = x1;
    a.y = y1;
    if (a.width <= 0 || a.height <= 0)
        a = Rect_<_Tp>();
    return a;
}

template<typename _Tp>
static inline Rect_<_Tp>& operator|=(Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    _Tp x1 = std::min(a.x, b.x), y1 = std::min(a.y, b.y);
    a.width = std::max(a.x + a.width, b.x + b.width) - x1;
    a.height = std::max(a.y + a.height, b.y + b.height) - y1;
    a.x = x1;
    a.y = y1;
    return a;
}

template<typename _Tp>
static inline Rect_<_Tp> operator&(const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    Rect_<_Tp> c = a;
    return c &= b;
}

template<typename _Tp>
static inline Rect_<_Tp> operator|(const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    Rect_<_Tp> c = a;
    return c |= b;
}

typedef Rect_<int> Rect;
typedef Rect_<float> Rect2f;

#define CV_8UC1  1
#define CV_8UC3  3
#define CV_8UC4  4
#define CV_32FC1 4

struct NCNN_EXPORT Mat
{
    Mat()
        : data(0), refcount(0), rows(0), cols(0), c(0)
    {
    }

    Mat(int _rows, int _cols, int flags)
        : data(0), refcount(0)
    {
        create(_rows, _cols, flags);
    }

    // copy
    Mat(const Mat& m)
        : data(m.data), refcount(m.refcount)
    {
        if (refcount)
            NCNN_XADD(refcount, 1);

        rows = m.rows;
        cols = m.cols;
        c = m.c;
    }

    Mat(int _rows, int _cols, int flags, void* _data)
        : data((unsigned char*)_data), refcount(0)
    {
        rows = _rows;
        cols = _cols;
        c = flags;
    }

    ~Mat()
    {
        release();
    }

    // assign
    Mat& operator=(const Mat& m)
    {
        if (this == &m)
            return *this;

        if (m.refcount)
            NCNN_XADD(m.refcount, 1);

        release();

        data = m.data;
        refcount = m.refcount;

        rows = m.rows;
        cols = m.cols;
        c = m.c;

        return *this;
    }

    Mat& operator=(const Scalar& s)
    {
        if (total() > 0)
        {
            uchar* p = data;
            for (int i = 0; i < cols * rows; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    *p++ = s[j];
                }
            }
        }

        return *this;
    }

    void create(int _rows, int _cols, int flags)
    {
        release();

        rows = _rows;
        cols = _cols;
        c = flags;

        if (total() > 0)
        {
            // refcount address must be aligned, so we expand totalsize here
            size_t totalsize = (total() + 3) >> 2 << 2;
            data = (uchar*)fastMalloc(totalsize + (int)sizeof(*refcount));
            refcount = (int*)(((uchar*)data) + totalsize);
            *refcount = 1;
        }
    }

    void release()
    {
        if (refcount && NCNN_XADD(refcount, -1) == 1)
            fastFree(data);

        data = 0;

        rows = 0;
        cols = 0;
        c = 0;

        refcount = 0;
    }

    Mat clone() const
    {
        if (empty())
            return Mat();

        Mat m(rows, cols, c);

        if (total() > 0)
        {
            memcpy(m.data, data, total());
        }

        return m;
    }

    bool empty() const
    {
        return data == 0 || total() == 0;
    }

    int channels() const
    {
        return c;
    }

    int type() const
    {
        return c;
    }

    size_t total() const
    {
        return cols * rows * c;
    }

    const uchar* ptr(int y) const
    {
        return data + y * cols * c;
    }

    uchar* ptr(int y)
    {
        return data + y * cols * c;
    }

    template<typename _Tp>
    const _Tp* ptr(int y) const
    {
        return (const _Tp*)data + y * cols * c;
    }

    template<typename _Tp>
    _Tp* ptr(int y)
    {
        return (_Tp*)data + y * cols * c;
    }

    // roi
    Mat operator()(const Rect& roi) const
    {
        if (empty())
            return Mat();

        Mat m(roi.height, roi.width, c);

        int sy = roi.y;
        for (int y = 0; y < roi.height; y++)
        {
            const uchar* sptr = ptr(sy) + roi.x * c;
            uchar* dptr = m.ptr(y);
            memcpy(dptr, sptr, roi.width * c);
            sy++;
        }

        return m;
    }

    uchar* data;

    // pointer to the reference counter;
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    int rows;
    int cols;

    int c;
};

enum ImreadModes
{
    IMREAD_UNCHANGED = -1,
    IMREAD_GRAYSCALE = 0,
    IMREAD_COLOR = 1
};

NCNN_EXPORT Mat imread(const std::string& path, int flags = IMREAD_COLOR);

enum ImwriteFlags
{
    IMWRITE_JPEG_QUALITY = 1
};

NCNN_EXPORT bool imwrite(const std::string& path, const Mat& m, const std::vector<int>& params = std::vector<int>());

NCNN_EXPORT void imshow(const std::string& name, const Mat& m);

NCNN_EXPORT int waitKey(int delay = 0);

// #if NCNN_PIXEL
NCNN_EXPORT void resize(const Mat& src, Mat& dst, const Size& size, float sw = 0.f, float sh = 0.f, int flags = 0);
// #endif // NCNN_PIXEL

// #if NCNN_PIXEL_DRAWING

enum
{
    FILLED = -1
};

NCNN_EXPORT void rectangle(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness = 1);

NCNN_EXPORT void rectangle(Mat& img, Rect rec, const Scalar& color, int thickness = 1);

NCNN_EXPORT void circle(Mat& img, Point center, int radius, const Scalar& color, int thickness = 1);

NCNN_EXPORT void line(Mat& img, Point p0, Point p1, const Scalar& color, int thickness = 1);

enum
{
    FONT_HERSHEY_SIMPLEX = 0
};

NCNN_EXPORT void putText(Mat& img, const std::string& text, Point org, int fontFace, double fontScale, Scalar color, int thickness = 1);

NCNN_EXPORT Size getTextSize(const std::string& text, int fontFace, double fontScale, int thickness, int* baseLine);

// #endif // NCNN_PIXEL_DRAWING

} // namespace cv

#if defined(_MSC_VER) || defined(__GNUC__)
#pragma pop_macro("min")
#pragma pop_macro("max")
#endif

// #endif // NCNN_SIMPLEOCV

#endif // NCNN_SIMPLEOCV_H
