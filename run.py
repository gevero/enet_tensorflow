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

    parser.add_argument('-sm',
                        '--save_model',
                        type=str,
                        default='./Enet.tf',
                        help='Tensorflow model save file')

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
                        default=8,
                        help='The batch size')

    parser.add_argument('-wd',
                        '--weight-decay',
                        type=float,
                        default=2e-4,
                        help='The weight decay')

    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=10,
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
                        default='./datasets/CamVid/train/images/*.png',
                        help='The path to the input dataset')

    parser.add_argument('-lptr',
                        '--label-pattern',
                        type=str,
                        default='./datasets/CamVid/train/labels/*.png',
                        help='The path to the label dataset')

    parser.add_argument(
        '-ipv',
        '--img-pattern-val',
        type=str,
        # default='./datasets/CamVid/val/images/*.png',
        default=
        './datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/*/*.png',
        help='The path to the input dataset')

    parser.add_argument(
        '-lpv',
        '--label-pattern-val',
        type=str,
        # default='./datasets/CamVid/val/labels/*.png',
        default=
        './datasets/cityscapes/gtFine_trainvaltest/gtFine/train/*/*labelIds*.png',
        help='The path to the label dataset')

    parser.add_argument('-iptt',
                        '--img-pattern-test',
                        type=str,
                        default='./datasets/CamVid/test/images/*.png',
                        help='The path to the input dataset')

    parser.add_argument('-lptt',
                        '--label-pattern-test',
                        type=str,
                        default='./datasets/CamVid/test/labels/*.png',
                        help='The path to the label dataset')

    parser.add_argument(
        '-ctr',
        '--cache-train',
        type=str,
        default='',
        help='Filename to cache the training data: if empty cache in memory')

    parser.add_argument(
        '-cv',
        '--cache-val',
        type=str,
        default='',
        help='Filename to cache the validation data: if empty cache in memory')

    parser.add_argument(
        '-ctt',
        '--cache-test',
        type=str,
        default='',
        help='Filename to cache the test data: if empty cache in memory')

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
