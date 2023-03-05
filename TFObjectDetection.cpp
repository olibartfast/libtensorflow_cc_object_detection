#include "TFObjectDetection.hpp"


float TFObjectDetection::compute_iou(const Detection& a, const Detection& b) {
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

vector<int> TFObjectDetection::ApplyNMS(const vector<Detection>& detections, float iou_threshold) {
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


tensorflow::Tensor TFObjectDetection::preprocess(const cv::Mat& frame)
{
    cv::Mat blob;
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
    return input_tensor;
}

vector<Detection> TFObjectDetection::infer(const cv::Mat& frame)
{
    auto input_tensor = preprocess(frame);

    // Run the inference
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        {"serving_default_input_tensor:0", input_tensor}
    };
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status status = session_->Run(inputs, {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1", "StatefulPartitionedCall:2", "StatefulPartitionedCall:3", "StatefulPartitionedCall:4"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << "Error running session: " << status.ToString() << "\n";
        std::exit(1);
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

    return output_detections;
}   