# Tensorflow object detection API inference with libtensorflow

Testing models from object detection API using Tensorflow c++ library.

##  Dependencies

* OpenCV  (tested 4.7.0)
* Libtensorflow_cc(tested 2.13, prebuilt library from [
Institut für Kraftfahrzeuge](https://github.com/ika-rwth-aachen/libtensorflow_cc) )

## Build 
```
cmake -E make_directory build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```
To build setting test and benchmarks OFF:
```
cmake -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF ..
```

##  Usage
```
./libtensorflow_cc_object_detection --saved_model_path=/path/to/saved_model/ --video_path=/path/to/video.mp4
```

|Model tested from [Tf2 model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)|
|-----------|
|ssd_resnet50_v1_fpn_640x640_coco17_tpu-8|
|ssd_mobilenet_v2_320x320_coco17_tpu-8|
|ssd_resnet101_v1_fpn_640x640_coco17_tpu-8|

To retrieve model info launch from command line: 

```
saved_model_cli show --dir /path/to/saved_model --tag_set serve --signature_def serving_default
```

If you don't have access to the saved_model_cli command-line tool, you can also load the saved model into a Python script and inspect the signature definition using the tf.saved_model.signature_constants module:

```
import tensorflow as tf

# Load the saved model
model = tf.saved_model.load('/path/to/saved_model')

# Get the signature definition
signature_def = model.signature_def[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

# Get the input tensor name and shape
input_tensor_name = signature_def.inputs['image_tensor'].name
input_tensor_shape = signature_def.inputs['image_tensor'].shape
print('Input tensor name: {}'.format(input_tensor_name))
print('Input tensor shape: {}'.format(input_tensor_shape))
```
