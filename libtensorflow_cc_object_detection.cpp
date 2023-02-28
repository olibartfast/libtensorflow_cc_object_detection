#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include "opencv2/opencv.hpp"

using namespace tensorflow;
using namespace std;

struct Detection {
    float ymin, xmin, ymax, xmax, score;
    int class_id;
};

float compute_iou(const Detection& a, const Detection& b) {
    // Calculate the coordinates of the intersection rectangle
    float xmin = std::max(a.xmin, b.xmin);
    float ymin = std::max(a.ymin, b.ymin);
    float xmax = std::min(a.xmax, b.xmax);
    float ymax = std::min(a.ymax, b.ymax);

    // Calculate the area of the intersection rectangle
    float intersection_area = std::max(0.0f, xmax - xmin) * std::max(0.0f, ymax - ymin);

    // Calculate the area of each bounding box
    float box1_area = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    float box2_area = (b.xmax - b.xmin) * (b.ymax - b.ymin);

    // Calculate the union area
    float union_area = box1_area + box2_area - intersection_area;

    // Calculate the IoU
    float iou = intersection_area / union_area;

    return iou;
}

vector<int> ApplyNMS(const vector<Detection>& detections, float iou_threshold) {
    vector<int> indices(detections.size());
    iota(indices.begin(), indices.end(), 0);

    // Sort the detections by score in descending order
    sort(indices.begin(), indices.end(),
         [&detections](int index1, int index2) {
             return detections[index1].score > detections[index2].score;
         });

    // Initialize the list of indices to keep
    vector<int> keep;

    // Loop over the sorted indices
    while (!indices.empty()) {
        int idx = indices[0];
        keep.push_back(idx);

        // Remove the current detection from the indices list
        indices.erase(indices.begin());

        // Compute IoU with all remaining detections
        vector<float> ious;
        for (int remaining_idx : indices) {
            float iou = compute_iou(detections[idx], detections[remaining_idx]);
            ious.push_back(iou);
        }

        // Remove detections with IoU above the threshold
        vector<int> iou_indices;
        for (int i = 0; i < ious.size(); ++i) {
            if (ious[i] > iou_threshold) {
                iou_indices.push_back(i);
            }
        }
        for (int i = iou_indices.size() - 1; i >= 0; --i) {
            int index = indices[iou_indices[i]];
            indices.erase(indices.begin() + iou_indices[i]);
        }
    }

    return keep;
}
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
  cv::Mat blob;
  while (cap.read(frame))
  {
   
    cv::cvtColor(frame, blob, cv::COLOR_BGR2RGB);
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
    status = session->Run(inputs, {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1", "StatefulPartitionedCall:2", "StatefulPartitionedCall:3", "StatefulPartitionedCall:4"}, {}, &outputs);
    if (!status.ok()) {
      std::cout << "Error running session: " << status.ToString() << "\n";
      return 1;
    }

        
    // Extract boxes, scores and class labels from the output tensors
    auto boxes = outputs[1].flat<float>();
    auto scores = outputs[4].flat<float>();
    auto classes = outputs[2].flat<float>();

    // Define threshold for detection score
    float detection_threshold = 0.5;

    // Define threshold for non-maximum suppression (NMS)
    float nms_threshold = 0.5;

    // Create vector to store detected objects
    std::vector<Detection> detections;

    // Iterate through all detected objects
    for (int i = 0; i < scores.size(); i++) {
        // Check if detection score is above the threshold
        if (scores(i) > detection_threshold) {
            // Extract the box coordinates
            int box_offset = i * 4;
            float ymin = boxes(box_offset) * frame.rows;
            float xmin = boxes(box_offset + 1) * frame.cols;
            float ymax = boxes(box_offset + 2) * frame.rows;
            float xmax = boxes(box_offset + 3) * frame.cols;

            // Extract the class label
            int class_id = static_cast<int>(classes(i));

            // Add the detection to the list
            detections.push_back({ymin, xmin, ymax, xmax, scores(i), class_id});
        }
    }

    vector<int> indices = ApplyNMS(detections, nms_threshold);

    // Collect the final list of detections
    vector<Detection> output_detections;
    for (int index : indices) {
        output_detections.push_back(detections[index]);
    }

    for(auto&& det : output_detections)
    {
      cv::rectangle(frame, cv::Rect(cv::Point(det.xmin, det.ymin), cv::Point(det.xmax, det.ymax)), cv::Scalar(0,0,255), 1);
      cv::putText(frame, std::to_string(det.class_id), cv::Point(det.xmin, det.ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));
    }
    
    cv::imshow("", frame);
    cv::waitKey(1);   
  }
  

  // Cleanup
  session->Close();
  return 0;
}

