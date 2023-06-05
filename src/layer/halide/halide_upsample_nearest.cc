#include "Halide.h"

class HalideUpsampleNearest : public Halide::Generator<HalideUpsampleNearest> {
public:
    Input<float> scale_x{"scale_x"};
    Input<float> scale_y{"scale_y"};

    Input<Halide::Buffer<float, 4>> input{"input"};

    Output<Halide::Buffer<float, 4>> output{"output"};

    Halide::Var b, y, x, c;

    void generate() {
        Halide::Expr x_scale =
            Halide::cast<int32_t>(Halide::cast<float>(x) * scale_x);
        Halide::Expr y_scale =
            Halide::cast<int32_t>(Halide::cast<float>(y) * scale_y);

        // TODO: schedule
        output(c, x, y, b) = input(c, x_scale, y_scale, b);
    }
};

HALIDE_REGISTER_GENERATOR(HalideUpsampleNearest, halide_upsample_nearest)
