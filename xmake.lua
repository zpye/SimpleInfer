add_rules("mode.debug", "mode.release")

set_languages("cxx17")

option("build_python")
    set_default(false)
    set_showmenu(true)
option_end()

option("halide")
    set_default(false)
    set_showmenu(true)
    add_includedirs("$(env HALIDE_ROOT)/include/", { public = true })
    add_linkdirs("$(env HALIDE_ROOT)/bin/Release/",
                 "$(env HALIDE_ROOT)/lib/Release/", { public = true })
    add_links("Halide")
    add_rpathdirs("$(env HALIDE_ROOT)/bin/Release/")
option_end()

includes("3rdparty")

if has_config("halide") then
    includes("src/layer/halide")
end

target("simple-infer")
    set_kind("static")
    add_includedirs("include/", { public = true })
    add_includedirs("src/")
    add_files("src/**.cpp")
    add_deps("eigen", "abseil-log", "cgraph", "highway")
    add_vectorexts("neon")
    add_vectorexts("sse", "sse2", "sse3", "ssse3")
    add_vectorexts("avx", "avx2")

    if has_config("halide") then
        add_deps("halide_layers")
    end

if has_config("build_python") then
    target("pybind11_export")
        before_build(function () 
            os.cp("python/simpleinfer/", "$(buildir)/python/")
        end)

        set_kind("shared")
        set_basename("simpleinfer")
        set_extension(".pyd")
        set_targetdir("$(buildir)/python/simpleinfer")
        add_files("python/pybind11_main.cpp")
        add_deps("pybind11", "simple-infer")

        after_build(function () 
            os.cp("python/setup.py.in", "$(buildir)/python/setup.py")
        end)
end

-- tests
target("test-eigen")
    set_kind("binary")
    add_includedirs("src/")
    add_files("test/test_3rdparty/test_eigen.cpp")
    add_deps("simple-infer")

target("test-im2col")
    set_kind("binary")
    add_includedirs("src/")
    add_files("test/test_3rdparty/test_im2col.cpp")
    add_deps("simple-infer")

target("test-broadcast")
    set_kind("binary")
    add_includedirs("src/")
    add_files("test/test_3rdparty/test_broadcast.cpp")
    add_deps("simple-infer")

target("test-engine")
    set_kind("binary")
    add_files("test/test_engine/test_engine.cpp")
    add_defines("MODEL_PATH=R\"($(curdir)/3rdparty/tmp)\"")
    add_deps("simple-infer")

target("test-pnnx-ir")
    set_kind("binary")
    add_includedirs("src/")
    add_files("test/test_pnnx/test_pnnx_ir.cpp")
    add_defines("MODEL_PATH=R\"($(curdir)/3rdparty/tmp)\"")
    add_deps("simple-infer")

target("test-layer")
    set_kind("binary")
    add_includedirs("src/", "test/")
    add_files("test/test_main.cpp")
    add_files("test/test_layer/**.cpp")
    add_deps("simple-infer", "catch2")

target("test-gemm")
    set_kind("binary")
    add_includedirs("src/", "test/")
    add_files("test/test_main.cpp")
    add_files("test/test_3rdparty/test_gemm.cpp")
    add_deps("simple-infer", "catch2")

target("test-yolo")
    set_kind("binary")
    add_includedirs("src/")
    add_files("test/test_yolo/test_yolo.cpp")
    add_defines("MODEL_PATH=R\"($(curdir)/3rdparty/tmp)\"")
    add_defines("IMAGE_PATH=R\"($(curdir)/imgs)\"")
    add_deps("simple-infer", "simpleocv")

target("test-yolo2")
    set_kind("binary")
    add_includedirs("src/")
    add_files("test/test_yolo/test_yolo2.cpp")
    add_defines("MODEL_PATH=R\"($(curdir)/3rdparty/tmp)\"")
    add_defines("IMAGE_PATH=R\"($(curdir)/imgs)\"")
    add_deps("simple-infer")

target("test-classify")
    set_kind("binary")
    add_includedirs("src/")
    add_files("test/test_classify/test_classify.cpp")
    add_defines("MODEL_PATH=R\"($(curdir)/3rdparty/tmp)\"")
    add_defines("IMAGE_PATH=R\"($(curdir)/imgs)\"")
    add_deps("simple-infer")

target("test-highway")
    set_kind("binary")
    add_includedirs("src/", "test/")
    add_files("src/logger.cpp")
    add_files("test/test_highway/test_highway.cpp")
    add_deps("abseil-log", "highway")

-- benchmark
target("bench")
    set_kind("binary")
    add_includedirs("src/")
    add_files("bench/**.cpp")
    add_defines("MODEL_PATH=R\"($(curdir)/3rdparty/tmp)\"")
    add_deps("simple-infer", "benchmark")
