# Caffe SegNet
**This is a modified version of [Caffe](https://github.com/BVLC/caffe) which supports the [SegNet architecture](http://mi.eng.cam.ac.uk/projects/segnet/)**

As described in **SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation** Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla [http://arxiv.org/abs/1511.00561]

## Getting Started with Example Model and Webcam Demo

If you would just like to try out a pretrained example model, then you can find the model used in the [SegNet webdemo](http://mi.eng.cam.ac.uk/projects/segnet/) and a script to run a live webcam demo here:
https://github.com/alexgkendall/SegNet-Tutorial

For a more detailed introduction to this software please see the tutorial here:
http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html

### Dataset

Prepare a text file of space-separated paths to images (jpegs or pngs) and corresponding label images alternatively e.g. ```/path/to/im1.png /another/path/to/lab1.png /path/to/im2.png /path/lab2.png ...```

Label images must be single channel, with each value from 0 being a separate class. The example net uses an image size of 360 by 480.

### Net specification

Example net specification and solver prototext files are given in examples/segnet.
To train a model, alter the data path in the ```data``` layers in ```net.prototxt``` to be your dataset.txt file (as described above).

In the last convolution layer, change ```num_output``` to be the number of classes in your dataset.

### Training

In solver.prototxt set a path for ```snapshot_prefix```. Then in a terminal run
```./build/tools/caffe train -solver ./examples/segnet/solver.prototxt```

## Publications

If you use this software in your research, please cite our publications:

http://arxiv.org/abs/1511.02680
Alex Kendall, Vijay Badrinarayanan and Roberto Cipolla "Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding." arXiv preprint arXiv:1511.02680, 2015.

http://arxiv.org/abs/1511.00561
Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." arXiv preprint arXiv:1511.00561, 2015. 


## License

This extension to the Caffe library is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here:
http://creativecommons.org/licenses/by-nc/4.0/
