/*
 * Simd Library (http://ermig1979.github.io/Simd).
 *
 * Copyright (c) 2011-2021 Yermalayeu Ihar.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "binary.h"

#include <cassert>

#include "hwy/highway.h"

namespace hwy {
namespace HWY_NAMESPACE {

static const Full128<float> d;
static_assert(4 == Lanes(d), "Lanes(Full128<float>) should be 4");
using f32x4_t = VFromD<Full128<float>>;

void AddBiasNHWC(const float* bias, size_t spatial, size_t oc, float* dst) {
    size_t oc4 = oc / 4 * 4;

    for (size_t s = 0; s < spatial; ++s) {
        size_t c = 0;
        for (; c < oc4; c += 4) {
            StoreU(Add(LoadU(d, dst + c), LoadU(d, bias + c)), d, dst + c);
        }

        for (; c < oc; ++c) {
            dst[c] += bias[c];
        }

        dst += oc;
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy

namespace SimpleInfer {

namespace hn = hwy::HWY_NAMESPACE;

void AddBiasNHWC(const float* bias, size_t spatial, size_t oc, float* dst) {
    return hn::AddBiasNHWC(bias, spatial, oc, dst);
}

}  // namespace SimpleInfer
