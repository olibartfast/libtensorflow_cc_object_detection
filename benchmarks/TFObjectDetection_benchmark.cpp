#include <benchmark/benchmark.h>
#include "TFObjectDetection.hpp"

static void BM_TFObjectDetection(benchmark::State& state, const char* model_path, const char* image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    TFObjectDetection detector(model_path);
    if (image.empty()) {
        std::cout << "Error loading image: " << image_path << std::endl;
        std::exit(1);
    }

    for (auto _ : state) {
        auto detections = detector.infer(image);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <model_path> <image_path>" << endl;
        return 1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];

    benchmark::RegisterBenchmark("BM_TFObjectDetection", BM_TFObjectDetection, model_path, image_path);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}