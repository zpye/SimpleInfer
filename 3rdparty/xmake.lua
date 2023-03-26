-- 3rdparty libs
-- set_languages("cxx20")

target("abseil-log")
    set_kind("static")
    add_defines("ABSL_BUILD_DLL=1")
    add_defines("ABSL_CONSUME_DLL=1", "NOMINMAX", { public = true })
    add_includedirs("abseil-cpp/", { public = true })
    add_files("abseil-cpp/absl/log/*.cc|*_test.cc|*_benchmark.cc|*_mock_*.cc",
              "abseil-cpp/absl/log/internal/*.cc|*_test.cc|test_*.cc",
              "abseil-cpp/absl/strings/**.cc|**_test.cc|**_benchmark.cc",
              "abseil-cpp/absl/time/**.cc|**_test.cc|**_benchmark.cc",
              "abseil-cpp/absl/flags/**.cc|**_test.cc|**_benchmark.cc",
              "abseil-cpp/absl/container/**.cc|**_test.cc|**_benchmark.cc",
              "abseil-cpp/absl/synchronization/**.cc|**_test.cc|**_benchmark.cc",
              "abseil-cpp/absl/debugging/**.cc|**_test.cc|**_benchmark.cc",
              "abseil-cpp/absl/numeric/**.cc|**_test.cc|**_benchmark.cc",
              "abseil-cpp/absl/crc/**.cc|**_test.cc|**_benchmark.cc",
              "abseil-cpp/absl/hash/**.cc|**_test.cc|**_benchmark.cc",
              "abseil-cpp/absl/base/**.cc|**_test.cc|**_testing.cc|**_test_common.cc|**_benchmark.cc")

target("cgraph")
    set_kind("static")
    add_includedirs("CGraph/src/", { public = true })
    add_files("CGraph/src/**.cpp")

target("eigen")
    set_kind("headeronly")
    add_includedirs("eigen/", { public = true })
    add_headerfiles("eigen/unsupported/Eigen/CXX11/**.h")
    add_defines("EIGEN_USE_THREADS", { public = true })

target("catch2")
    set_kind("headeronly")
    add_includedirs("Catch2/single_include/", { public = true })
    add_headerfiles("Catch2/single_include/catch2/catch.hpp")

target("benchmark")
    set_kind("static")
    add_rules("mode.release", "mode.debug")
    add_includedirs("benchmark/include/", { public = true })
    add_files("benchmark/src/**.cc")
    add_defines("BENCHMARK_STATIC_DEFINE", { public = true })
    set_optimize("fastest")
    if is_plat("windows") then
        add_syslinks("shlwapi")
    end
