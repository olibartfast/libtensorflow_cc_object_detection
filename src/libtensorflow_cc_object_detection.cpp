#include "TFObjectDetection.hpp"

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

  std::cout<< "Video: " << video_path << std::endl;
  std::cout << "Model: " << saved_model_path << std::endl;

  // Create a TFObjectDetection instance with the path to the saved model
  TFObjectDetection detector(saved_model_path);
  
  cv::VideoCapture cap(video_path);
  cv::Mat frame;
  cv::Mat blob;
  while (cap.read(frame))
  {
    auto output_detections = detector.infer(frame);
    for(auto&& det : output_detections)
    {
      cv::rectangle(frame, cv::Rect(cv::Point(det.xmin, det.ymin), cv::Point(det.xmax, det.ymax)), cv::Scalar(0,0,255), 1);
      cv::putText(frame, std::to_string(det.class_id), cv::Point(det.xmin, det.ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));
    }
    
    cv::imshow("", frame);
    cv::waitKey(1);   
  }
  

  return 0;
}

