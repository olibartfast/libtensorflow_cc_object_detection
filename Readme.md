# Tensorflow object detection API inference with libtensorflow

Testing models from object detection API using Tensorflow c++ library, model tested in this repo: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8

##  Dependencies

* OpenCV  (tested 4.6.0)
* Libtensorflow_cc(tested 2.11, prebuilt library from [
Institut für Kraftfahrzeuge](https://github.com/ika-rwth-aachen/libtensorflow_cc) )

## Build 
```
cmake -E make_directory build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

##  Usage
```
./libtensorflow_cc_object_detection --saved_model_path=/path/to/saved_model/ --video_path=/path/to/video.mp4
```