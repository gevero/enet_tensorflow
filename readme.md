# ENet

This repository contains a tensorflow 2.0 implementation of Enet as in: 

Paszke, A.; Chaurasia, A.; Kim, S.; Culurciello, E. ENet: **A Deep Neural Network Architecture for Real-Time Semantic Segmentation.** [arXiv:1606.02147 [cs] 2016.](https://arxiv.org/pdf/1606.02147.pdf)

 Enet is a lightweight neural network geared towards image segmentation for real time applications. This tensorflow 2.0 implementation is greatly indebted with the PyTorch [ENet - Real Time Semantic Segmentation](https://github.com/iArunava/ENet-Real-Time-Semantic-Segmentation) implementation by [iArunava](https://github.com/iArunava). 

# Try it out

You can try it out directly in this [Colab notebook](https://colab.research.google.com/github/gevero/enet_tensorflow/blob/master/notebooks/Enet%20CamVid%20Training.ipynb). In the notebook, Enet is trained in three different ways for comparison:

- **First the Encoder and then the Decoder:** as in the original paper, we first train the encoder, freeze the weights and then train the decoder. This approach provides the more stable training.
-  **Encoder and  Decoder simultaneously with two objectives:** an approach similar to the original paper: we train Enet in one go but with the stabilty benefits of the original approach.
-  **End to End:** we train Enet in one go. It is the least stable method, albeit the simplest.

## Pretrained weights

You can find them [here](https://drive.google.com/open?id=1rQN_855G-iHZkPe7KEI-P5PF8U4uIf40) for the [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset. The weights for different datasets will be released as soon as possible.

## A typical example
![TestImg](https://github.com/gevero/enet_tensorflow/blob/master/Images/SegmentationExample.png)

