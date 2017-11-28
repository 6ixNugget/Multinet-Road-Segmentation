# Multinet-Road-Segmentation

This is my attempt of implementing the semantic segmentation part of Multinet, which is a simplified structure similar to Segnet, with Keras.

The goal was to familiarize myself with Keras libarary while implementing a small project using a fairly complex(deep) neural network.

- The code was tested on the KITTI Vision Benchmark Suite road dataset. Find it [here](http://www.cvlibs.net/datasets/kitti/eval_road.php).
- Stack: Python 3.5 Tensorflow 1.3.0 

### Training
When running for the first time, the script will automatically download the dataset and extract the compressed .zip file
```sh
$ python3 keras.py
```
After each run the script will save a trained model into the current folder under the name *trained_model.h5*.
Please refer to [Keras FAQ](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) about how to load and continue training on the saved model

### Structure
The network has an encoder-decoder structure similar with the one pioneered by [SegNet](https://arxiv.org/abs/1511.00561). It is composed of the first 13 conv layers of VGG16 (without the dense layers) as the encoder, and a semantic segmentation decoder following the FCN architecture. Given the encoder, the remaining fully connected (FC) layers of the VGG architecture are transformed into 1 × 1 convolutional layers to produce a low resolution segmentations. It is then followed by three transposed conv layers to perform upsampling. Notice that there are skip connections from lower layers to extract high resolution features. Those features are topped by 1 × 1 conv layers and then added into the upsampled results from transposed conv layers.

![alt text](https://s3.us-east-2.amazonaws.com/hosted-downloadable-files/Multinet+road+seg+structure.png "structure visual")
