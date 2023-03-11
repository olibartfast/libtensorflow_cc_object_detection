#include <gtest/gtest.h>
#include "TFObjectDetection.hpp"

using namespace testing;

class TFObjectDetectionTest : public ::testing::Test {
protected:
    std::string model_path_;
    float score_threshold_ = 0.5;
    std::string image_path_; // declare image_path_

    virtual void SetUp() {
        // set up the TFObjectDetection object here
        object_detection_ = new TFObjectDetection(model_path_, score_threshold_);
    }

    virtual void TearDown() {
        // tear down the TFObjectDetection object here
        delete object_detection_;
    }

    // declare any variables or pointers needed for the test cases
    TFObjectDetection* object_detection_;

public:
    void SetImagePath(const std::string& image_path) {
        image_path_ = image_path; 
    }
    void  SetModelPath(const std::string& model_path) {
        model_path_ = model_path;
    }
    
    // Implement the pure virtual function
    void TestBody() override {}
};


TEST_F(TFObjectDetectionTest, TestInfer) {
    // set up the input frame
    cv::Mat frame = cv::imread(image_path_, cv::IMREAD_COLOR);

    // call the infer function and get the detections
    std::vector<Detection> detections = object_detection_->infer(frame);

    // check that the detections are not empty
    EXPECT_FALSE(detections.empty());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (argc < 3) { // expect three arguments: the executable name, model_path, and image_path
        std::cerr << "Usage: " << argv[0] << " path/to/model path/to/image\n";
        return 1;
    }
    std::string model_path = argv[1];
    std::string image_path = argv[2];

    TFObjectDetectionTest test;
    test.SetModelPath(model_path);
    test.SetImagePath(image_path);

    return RUN_ALL_TESTS();
}