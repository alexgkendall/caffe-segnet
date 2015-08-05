# Caffe SegNet
**This is a modified version of caffe(https://github.com/BVLC/caffe) which supports the SegNet architecture**

As described in **SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling** Vijay Badrinarayanan, Ankur Handa, Roberto Cipolla [http://arxiv.org/abs/1505.07293]

## Usage

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

Though in the paper SegNet is trained with a layer-wise LBFGS method, here we train all layers simulataneously using ADAGRAD.
