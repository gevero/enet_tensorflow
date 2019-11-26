import numpy as np
import argparse
from train import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        type=str,
                        default='./datasets/CamVid/ckpt-camvid-enet.pth',
                        help='The path to the pretrained enet model')

    parser.add_argument(
        '-i',
        '--image-path',
        type=str,
        help='The path to the image to perform semantic segmentation')

    parser.add_argument('-tbl',
                        '--tensorboard_logs',
                        type=str,
                        default='./tb_logs/',
                        help='Tensorboard logs folder')

    parser.add_argument('-tt',
                        '--training-type',
                        type=int,
                        default=1,
                        help='0: end->dec, 1: end+dec, 2: dec')

    parser.add_argument('-ih',
                        '--img-height',
                        type=int,
                        default=360,
                        help='The height for the resized image')

    parser.add_argument('-iw',
                        '--img-width',
                        type=int,
                        default=480,
                        help='The width for the resized image')

    parser.add_argument('-lr',
                        '--learning-rate',
                        type=float,
                        default=5e-4,
                        help='The learning rate')

    parser.add_argument('-bs',
                        '--batch-size',
                        type=int,
                        default=10,
                        help='The batch size')

    parser.add_argument('-wd',
                        '--weight-decay',
                        type=float,
                        default=2e-4,
                        help='The weight decay')

    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=102,
                        help='The number of epochs')

    parser.add_argument('-nc',
                        '--num-classes',
                        type=int,
                        default=12,
                        help='The number of epochs')

    parser.add_argument(
        '-se',
        '--save-every',
        type=int,
        default=10,
        help='The number of epochs after which to save a model')

    parser.add_argument('-iptr',
                        '--img-pattern',
                        type=str,
                        default='./datasets/CamVid/train/img/',
                        help='The path to the input dataset')

    parser.add_argument('-lptr',
                        '--label-pattern',
                        type=str,
                        default='./datasets/CamVid/train/labels/',
                        help='The path to the label dataset')

    parser.add_argument('-ipv',
                        '--img-pattern-val',
                        type=str,
                        default='./datasets/CamVid/val/img/',
                        help='The path to the input dataset')

    parser.add_argument('-lpv',
                        '--label-pattern-val',
                        type=str,
                        default='./datasets/CamVid/val/labels/',
                        help='The path to the label dataset')

    parser.add_argument('-iptt',
                        '--img-pattern-test',
                        type=str,
                        default='./datasets/CamVid/test/img/',
                        help='The path to the input dataset')

    parser.add_argument('-lptt',
                        '--label-pattern-test',
                        type=str,
                        default='./datasets/CamVid/test/labels',
                        help='The path to the label dataset')

    parser.add_argument('--mode',
                        choices=['train', 'test'],
                        default='train',
                        help='Whether to train or test')

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.mode.lower() == 'train':
        train(FLAGS)
    elif FLAGS.mode.lower() == 'test':
        test(FLAGS)
    else:
        raise RuntimeError(
            'Unknown mode passed. \n Mode passed should be either \
                            of "train" or "test"')
