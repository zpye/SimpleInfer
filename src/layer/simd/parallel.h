/*
 * Simd Library (http://ermig1979.github.io/Simd).
 *
 * Copyright (c) 2011-2023 Yermalayeu Ihar.
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

#ifndef SIMPLE_INFER_SRC_LAYER_SIMD_PARALLEL_H_
#define SIMPLE_INFER_SRC_LAYER_SIMD_PARALLEL_H_

#include <future>
#include <thread>
#include <vector>

namespace SimpleInfer {

// void(thread, begin, end)
template<typename Function>
inline void Parallel(size_t begin,
                     size_t end,
                     const Function& function,
                     size_t thread_number,
                     size_t block_align = 1) {
    static const size_t thread_number_max = std::thread::hardware_concurrency();
    thread_number = (std::min)(thread_number, thread_number_max);

    if (thread_number <= 1 || size_t(block_align * 1.5) >= (end - begin)) {
        function(0, begin, end);
    } else {
        std::vector<std::future<void>> futures;

        size_t block_size = (end - begin + thread_number - 1) / thread_number;
        block_size = (block_size + block_align - 1) / block_align * block_align;

        size_t block_begin = begin;
        size_t block_end   = block_begin + block_size;

        for (size_t thread = 0; thread < thread_number && block_begin < end;
             ++thread) {
            futures.push_back(std::move(
                std::async(std::launch::async,
                           [block_begin, block_end, thread, &function] {
                               function(thread, block_begin, block_end);
                           })));
            block_begin += block_size;
            block_end = (std::min)(block_begin + block_size, end);
        }

        for (size_t i = 0; i < futures.size(); ++i) {
            futures[i].wait();
        }
    }
}

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_SIMD_PARALLEL_H_
