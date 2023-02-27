#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include "opencv2/opencv.hpp"

using namespace tensorflow;
using namespace std;

static const std::string params = "{ help h   |   | print help message }"
      "{ saved_model_path s     |  | path to saved model}"
      "{ video_path v   |   | path to input video source}";


int main (int argc, char *argv[])
{
  // Command line parser
  cv::CommandLineParser parser(argc, argv, params);
  parser.about("Object detection api infer with libtensorflow c++");
  if (parser.has("help")){
    parser.printMessage();
    return 0;  
  }

  const std::string saved_model_path = parser.get<std::string>("saved_model_path");
  const std::string video_path = parser.get<std::string>("video_path");

  if (saved_model_path.empty()){
      std::cerr << "Missing saved model path" << std::endl;
      std::exit(1);
  }

  if (video_path.empty()){
    std::cerr << "Missing video path" << std::endl;
    std::exit(1);
  }



  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;



  Status status = LoadSavedModel(session_options, run_options, 
  saved_model_path, {kSavedModelTagServe}, &bundle);
  if (!status.ok()) {
    std::cout << "Error loading SavedModel: " << status.ToString() << "\n";
    return 1;
  }

  // Create a new session and attach the graph
  Session* session = bundle.session.get();

  cv::VideoCapture cap(video_path);

  cv::Mat frame;
  while (cap.read(frame))
  {
    cv::Mat blob = frame;
    // Convert the frame to a TensorFlow tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_UINT8, tensorflow::TensorShape({1, blob.rows, blob.cols, blob.channels()}));
    auto input_tensor_mapped = input_tensor.tensor<uint8_t, 4>();

    // Copy the data from the input frame to the input tensor
    for (int y = 0; y < blob.rows; ++y) {
      const uchar* row_ptr = blob.ptr<uchar>(y);
      for (int x = 0; x < blob.cols; ++x) {
        const uchar* pixel_ptr = row_ptr + (x * blob.channels());
        for (int c = 0; c < blob.channels(); ++c) {
          input_tensor_mapped(0, y, x, c) = static_cast<uint8_t>(pixel_ptr[c]);
        }
      }
    }
    // Run the inference
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        {"serving_default_input_tensor:0", input_tensor}
    };
    std::vector<tensorflow::Tensor> outputs;
    status = session->Run(inputs, {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1", "StatefulPartitionedCall:2", "StatefulPartitionedCall:3"}, {}, &outputs);
    if (!status.ok()) {
      std::cout << "Error running session: " << status.ToString() << "\n";
      return 1;
    }
    auto boxes = outputs[1].tensor<float, 3>();

    auto output_tensor = boxes;
    for (int i = 0; i < output_tensor.dimension(1); i++) {
        float score = output_tensor(0, i, 4);
        if (score < 0.5) break;
        int x_min = static_cast<int>(output_tensor(0, i, 1) * frame.cols);
        int y_min = static_cast<int>(output_tensor(0, i, 0) * frame.rows);
        int x_max = static_cast<int>(output_tensor(0, i, 3) * frame.cols);
        int y_max = static_cast<int>(output_tensor(0, i, 2) * frame.rows);
        cv::rectangle(frame, cv::Rect(cv::Point(x_min, y_min), cv::Point(x_max, y_max)), cv::Scalar(0,0,255), 1);
        cout << "Detected object with score " << score << " at (" << x_min << ", " << y_min << ") - (" << x_max << ", " << y_max << ")" << endl;
    }
    cv::imshow("", frame);
    cv::waitKey(1);

    

  }
  

  // Cleanup
  session->Close();
  return 0;
}

