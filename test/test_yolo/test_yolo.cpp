#include <cfloat>
#include <string>

#include "engine.h"
#include "logger.h"

#include "simpleocv.h"

using namespace SimpleInfer;

// from ncnn
struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct Adjust {
    int padding_l = 0;
    int padding_t = 0;
    float scale   = 1.0f;
};

static inline float intersection_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects,
                                  int left,
                                  int right) {
    int i   = left;
    int j   = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j)
                qsort_descent_inplace(faceobjects, left, j);
        }
        // #pragma omp section
        {
            if (i < right)
                qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects,
                              std::vector<int>& picked,
                              float nms_threshold,
                              bool agnostic = false) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void draw_objects(const cv::Mat& bgr,
                         const std::vector<Object>& objects,
                         cv::Mat& result) {
    static const char* class_names[] = {
        "person",        "bicycle",      "car",
        "motorcycle",    "airplane",     "bus",
        "train",         "truck",        "boat",
        "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench",        "bird",
        "cat",           "dog",          "horse",
        "sheep",         "cow",          "elephant",
        "bear",          "zebra",        "giraffe",
        "backpack",      "umbrella",     "handbag",
        "tie",           "suitcase",     "frisbee",
        "skis",          "snowboard",    "sports ball",
        "kite",          "baseball bat", "baseball glove",
        "skateboard",    "surfboard",    "tennis racket",
        "bottle",        "wine glass",   "cup",
        "fork",          "knife",        "spoon",
        "bowl",          "banana",       "apple",
        "sandwich",      "orange",       "broccoli",
        "carrot",        "hot dog",      "pizza",
        "donut",         "cake",         "chair",
        "couch",         "potted plant", "bed",
        "dining table",  "toilet",       "tv",
        "laptop",        "mouse",        "remote",
        "keyboard",      "cell phone",   "microwave",
        "oven",          "toaster",      "sink",
        "refrigerator",  "book",         "clock",
        "vase",          "scissors",     "teddy bear",
        "hair drier",    "toothbrush"};

    result = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];

        LOG(INFO) << absl::StrFormat(
            "%d = %.5f at (%.2f, %.2f) size(%.2f x %.2f)",
            obj.label,
            obj.prob,
            obj.rect.x,
            obj.rect.y,
            obj.rect.width,
            obj.rect.height);

        cv::rectangle(result, obj.rect, cv::Scalar(255, 0, 0), 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size =
            cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > result.cols)
            x = result.cols - label_size.width;

        cv::rectangle(
            result,
            cv::Rect(cv::Point(x, y),
                     cv::Size(label_size.width, label_size.height + baseLine)),
            cv::Scalar(255, 255, 255),
            -1);

        cv::putText(result,
                    text,
                    cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 0, 0));
    }
}

template<typename T>
T clip(const T& n, const T& lower, const T& upper) {
    return (std::max)(lower, (std::min)(n, upper));
}

EigenTensor<float, 4> PreProcess(const cv::Mat& input_image,
                                 const int height_new,
                                 const int width_new,
                                 Adjust& adjust) {
    const int height_origin = input_image.rows;
    const int width_origin  = input_image.cols;

    int height_resize = height_new;
    int width_resize  = width_new;

    // keep ratio of h and w
    float scale = 1.0f;
    if (height_new * width_origin < width_new * height_origin) {
        scale        = (float)height_new / (float)height_origin;
        width_resize = (int)(width_origin * scale);
    } else {
        scale         = (float)width_new / (float)width_origin;
        height_resize = (int)(height_origin * scale);
    }

    cv::Mat resize_image;
    cv::resize(input_image,
               resize_image,
               cv::Size(width_resize, height_resize));

    EigenTensorMap<uchar, 4> resize_image_eigen_tensor(
        resize_image.data,
        EigenDSize<4>(1, height_resize, width_resize, 3));

    // bgr -> rgb
    std::array<bool, 4> reverse_dim;
    reverse_dim[0] = false;
    reverse_dim[1] = false;
    reverse_dim[2] = false;
    reverse_dim[3] = true;

    auto expr_reverse = resize_image_eigen_tensor.reverse(reverse_dim);

    // padding
    int padding_h = height_new - height_resize;
    int padding_w = width_new - width_resize;

    int padding_t = padding_h / 2;
    int padding_b = padding_h - padding_t;
    int padding_l = padding_w / 2;
    int padding_r = padding_w - padding_l;

    std::array<std::pair<ptrdiff_t, ptrdiff_t>, 4> paddings;
    paddings[0] = std::make_pair(0, 0);
    paddings[1] = std::make_pair(padding_t, padding_b);
    paddings[2] = std::make_pair(padding_l, padding_r);
    paddings[3] = std::make_pair(0, 0);

    auto expr_padding = expr_reverse.pad(paddings, 114);

    // cast
    auto expr_cast = expr_padding.cast<float>();

    // normalize
    auto expr_normalize = expr_cast / 255.0f;

    adjust.scale     = scale;
    adjust.padding_l = padding_l;
    adjust.padding_t = padding_t;

    return EigenTensor<float, 4>(expr_normalize);
}

cv::Mat ToImage(const EigenTensor<float, 4>& eigen_tensor) {
    // rgb -> bgr
    std::array<bool, 4> reverse_dim;
    reverse_dim[0] = false;
    reverse_dim[1] = false;
    reverse_dim[2] = false;
    reverse_dim[3] = true;

    EigenTensor<uchar, 4> img_eigen_tensor =
        (eigen_tensor * 255.0f).cast<uchar>().reverse(reverse_dim);

    EigenDSize<4> dsize = img_eigen_tensor.dimensions();

    return cv::Mat(dsize[2], dsize[1], dsize[3], img_eigen_tensor.data())
        .clone();
}

int main() {
    InitializeContext();

    const std::string model_path(MODEL_PATH);
    const std::string param_file =
        model_path + "/yolo/demo/yolov5s_batch4.pnnx.param";
    const std::string bin_file =
        model_path + "/yolo/demo/yolov5s_batch4.pnnx.bin";
    int input_height = 640;
    int input_width  = 640;
    // const std::string param_file =
    //     model_path + "/yolo/demo/yolov5n_small.pnnx.param";
    // const std::string bin_file =
    //     model_path + "/yolo/demo/yolov5n_small.pnnx.bin";
    // int input_height = 320;
    // int input_width  = 320;

    Engine engine;
    engine.LoadModel(param_file, bin_file);

    // set input image data
    Tensor input(DataType::kFloat32, {4, input_height, input_width, 3}, true);
    EigenTensorMap<float, 4> input_eigen_tensor =
        input.GetEigenTensor<float, 4>();

    const std::string image_path(IMAGE_PATH);
    const std::string image_names[4] = {"31.jpg",
                                        "bus.jpg",
                                        "car.jpg",
                                        "zidane.jpg"};
    cv::Mat images[4];
    Adjust adjusts[4];

    for (int i = 0; i < 4; ++i) {
        const std::string image_file = image_path + "/" + image_names[i];

        images[i] = cv::imread(image_file, cv::ImreadModes::IMREAD_COLOR);

        EigenTensor<float, 4> img_eigen_tensor =
            PreProcess(images[i], input_height, input_width, adjusts[i]);

        input_eigen_tensor.slice(EigenDSize<4>(i, 0, 0, 0),
                                 img_eigen_tensor.dimensions()) =
            img_eigen_tensor;

        // save padding image
        // cv::imwrite("output_" + image_names[i], ToImage(img_eigen_tensor));
    }

    // inference
    engine.Input("0", input);
    engine.Forward();

    Tensor output;
    engine.Extract("140", output);

    {
        std::vector<int> output_shape = output.Shape();
        std::string shape_str;
        for (auto& d : output_shape) {
            shape_str = absl::StrAppendFormat(&shape_str, " %d", d);
        }
        LOG(INFO) << "output shape:" << shape_str;
    }

    EigenTensorMap<float, 3> result = output.GetEigenTensor<float, 3>();
    EigenDSize<3> result_shape      = result.dimensions();

    const int elements  = result_shape[1];
    const int num_class = result_shape[2] - 5;

    // get detection boxes
    const float prob_threshold = 0.25f;
    const float nms_threshold  = 0.45f;
    for (int b = 0; b < 4; ++b) {  // result_shape[0]
        std::vector<Object> objects;
        for (int e = 0; e < elements; ++e) {
            float box_score = result(b, e, 4);

            // find class index with max class score
            int class_index   = -1;
            float class_score = -FLT_MAX;
            for (int k = 0; k < num_class; ++k) {
                float score = result(b, e, k + 5);
                if (score > class_score) {
                    class_index = k;
                    class_score = score;
                }
            }

            float confidence = box_score * class_score;

            if (confidence >= prob_threshold) {
                float pb_cx = result(b, e, 0);
                float pb_cy = result(b, e, 1);
                float pb_w  = result(b, e, 2);
                float pb_h  = result(b, e, 3);

                float x0 = pb_cx - pb_w * 0.5f;
                float y0 = pb_cy - pb_h * 0.5f;
                float x1 = pb_cx + pb_w * 0.5f;
                float y1 = pb_cy + pb_h * 0.5f;

                Object obj;
                obj.rect.x      = x0;
                obj.rect.y      = y0;
                obj.rect.width  = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label       = class_index;
                obj.prob        = confidence;

                objects.push_back(obj);
            }
        }

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(objects);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(objects, picked, nms_threshold);

        int count = picked.size();

        std::vector<Object> objects_result(count);
        for (int i = 0; i < count; i++) {
            objects_result[i] = objects[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects_result[i].rect.x - adjusts[b].padding_l) /
                       adjusts[b].scale;
            float y0 = (objects_result[i].rect.y - adjusts[b].padding_t) /
                       adjusts[b].scale;
            float x1 = (objects_result[i].rect.x +
                        objects_result[i].rect.width - adjusts[b].padding_l) /
                       adjusts[b].scale;
            float y1 = (objects_result[i].rect.y +
                        objects_result[i].rect.height - adjusts[b].padding_t) /
                       adjusts[b].scale;

            // clip
            x0 = clip(x0, 0.0f, (float)(images[b].cols - 1));
            y0 = clip(y0, 0.0f, (float)(images[b].rows - 1));
            x1 = clip(x1, 0.0f, (float)(images[b].cols - 1));
            y1 = clip(y1, 0.0f, (float)(images[b].rows - 1));

            objects_result[i].rect.x      = x0;
            objects_result[i].rect.y      = y0;
            objects_result[i].rect.width  = x1 - x0;
            objects_result[i].rect.height = y1 - y0;
        }

        cv::Mat result;
        draw_objects(images[b], objects_result, result);

        const std::string name = absl::StrFormat("%s/yolo_result_%s",
                                                 image_path.c_str(),
                                                 image_names[b].c_str());
        cv::imwrite(name, result);
        LOG(INFO) << "save image: " << name;
    }

    return 0;
}
