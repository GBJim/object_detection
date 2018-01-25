# Windows Branch of Tensorflow Object Detectors

## Prerequisite
Install the following dependencies sequentially

### 1.Nvidia GPU Drivier
1. Download the latest Nvidia Driver for GPU from Nvidia Website. [Download Link](http://www.nvidia.com.tw/Download/index.aspx)
2. Install the driver. 

### 2.CUDA Toolkit 8.0
1. Download CUDA Toolkit 8.0 for Windows from Nvidia Website. [Download Link](https://developer.nvidia.com/cuda-80-ga2-download-archive) 
2. Install the CUDA Toolkit. 

### 3.cuDNN V6
1. Download the Windows version of cuDNN v6 for CUDA 8.0 from Nvidia Website. [Download Link](https://developer.nvidia.com/cudnn)(You need to register for downloading)
2. Unzip the folders bin, lib, and include to the CUDA installation path, which is C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0 by default

### 4.Miniconda
1. Download the Miniconda of Python2.7 for Windows. [Download Link](https://conda.io/miniconda.html)
2. Install Miniconda
3. Add the conda path 'C:\ProgramData\Miniconda2' into the PATH system variable

### 5.TensorFlow
1. Launch the CMDt and install TensorFlow as following:

```
#Create a conda environment named tensorflow
C:> conda create -n tensorflow python=3.5
#Activate the conda environment
C:> activate tensorflow
 (tensorflow)C:>  # Your prompt should change  
#Install Tensorflow
(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow-gpu 
```

2. To validate the installation, do as following in CMD:

```
$ python
#After entering the python shell, run the following lines.

>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))

#You should see prompts about GPU messages
```

### 6.OpenCV and Python Dependencies 
1. In the conda environment, install openCV

```
(tensorflow)C:> conda install -c menpo opencv
```

2. Install all the required Python packages

```
(tensorflow)C:> pip install matplotlib pillow lxml
```

## Install Tensorflow:Object Detection
1. Clone this repository

```
$ git clone http://172.16.15.205/ainvr/object_detection.git
```

2. Download the protoc 3.4.0 binary for Windows. [Download Link](https://github.com/google/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip). Unzip the  bin/protoc.exe to the directory which contains the object_detection repo.

3. Compile the prototxt for the detection API

```
#From the corresponding directory 
$ protoc object_detection/protos/*.proto --python_out=.
```

4. Add a new environment variable PYTHONPATH, add the path of object_detection/slim

5. Download the trained <b>faster_rcnn_inception_v2_coco model</b>. [Download Link](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2017_11_08.tar.gz). Unzip it to the object_detection directory. 

6. Run a simple dmeo
```
#From the object_detection directory
$ python detector.py

```

# Tensorflow Object Detection API
Creating accurate machine learning models capable of localizing and identifying
multiple objects in a single image remains a core challenge in computer vision.
The TensorFlow Object Detection API is an open source framework built on top of
TensorFlow that makes it easy to construct, train and deploy object detection
models.  At Google we’ve certainly found this codebase to be useful for our
computer vision needs, and we hope that you will as well.
<p align="center">
  <img src="g3doc/img/kites_detections_output.jpg" width=676 height=450>
</p>
Contributions to the codebase are welcome and we would love to hear back from
you if you find this API useful.  Finally if you use the Tensorflow Object
Detection API for a research publication, please consider citing:

```
"Speed/accuracy trade-offs for modern convolutional object detectors."
Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z,
Song Y, Guadarrama S, Murphy K, CVPR 2017
```
\[[link](https://arxiv.org/abs/1611.10012)\]\[[bibtex](
https://scholar.googleusercontent.com/scholar.bib?q=info:l291WsrB-hQJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAWUIIlnPZ_L9jxvPwcC49kDlELtaeIyU-&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1)\]

## Maintainers

* Jonathan Huang, github: [jch1](https://github.com/jch1)
* Vivek Rathod, github: [tombstone](https://github.com/tombstone)
* Derek Chow, github: [derekjchow](https://github.com/derekjchow)
* Chen Sun, github: [jesu9](https://github.com/jesu9)
* Menglong Zhu, github: [dreamdragon](https://github.com/dreamdragon)


## Table of contents

Before You Start:
* <a href='g3doc/installation.md'>Installation</a><br>

Quick Start:
* <a href='object_detection_tutorial.ipynb'>
      Quick Start: Jupyter notebook for off-the-shelf inference</a><br>
* <a href="g3doc/running_pets.md">Quick Start: Training a pet detector</a><br>

Setup:
* <a href='g3doc/configuring_jobs.md'>
      Configuring an object detection pipeline</a><br>
* <a href='g3doc/preparing_inputs.md'>Preparing inputs</a><br>

Running:
* <a href='g3doc/running_locally.md'>Running locally</a><br>
* <a href='g3doc/running_on_cloud.md'>Running on the cloud</a><br>

Extras:
* <a href='g3doc/detection_model_zoo.md'>Tensorflow detection model zoo</a><br>
* <a href='g3doc/exporting_models.md'>
      Exporting a trained model for inference</a><br>
* <a href='g3doc/defining_your_own_model.md'>
      Defining your own model architecture</a><br>
* <a href='g3doc/using_your_own_dataset.md'>
      Bringing in your own dataset</a><br>

## Getting Help

Please report bugs to the tensorflow/models/ Github
[issue tracker](https://github.com/tensorflow/models/issues), prefixing the
issue name with "object_detection". To get help with issues you may encounter
using the Tensorflow Object Detection API, create a new question on
[StackOverflow](https://stackoverflow.com/) with the tags "tensorflow" and
"object-detection".

## Release information

### June 15, 2017

In addition to our base Tensorflow detection model definitions, this
release includes:

* A selection of trainable detection models, including:
  * Single Shot Multibox Detector (SSD) with MobileNet,
  * SSD with Inception V2,
  * Region-Based Fully Convolutional Networks (R-FCN) with Resnet 101,
  * Faster RCNN with Resnet 101,
  * Faster RCNN with Inception Resnet v2
* Frozen weights (trained on the COCO dataset) for each of the above models to
  be used for out-of-the-box inference purposes.
* A [Jupyter notebook](object_detection_tutorial.ipynb) for performing
  out-of-the-box inference with one of our released models
* Convenient [local training](g3doc/running_locally.md) scripts as well as
  distributed training and evaluation pipelines via
  [Google Cloud](g3doc/running_on_cloud.md).


<b>Thanks to contributors</b>: Jonathan Huang, Vivek Rathod, Derek Chow,
Chen Sun, Menglong Zhu, Matthew Tang, Anoop Korattikara, Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, Jasper Uijlings,
Viacheslav Kovalevskyi, Kevin Murphy