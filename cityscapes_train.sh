#!/bin/bash
IW=512
IH=1024
BS=16
EP=10
NC=35
IPTR="./datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/*/*.png"
LPTR="./datasets/cityscapes/gtFine_trainvaltest/gtFine/train/*/*labelIds*.png"
IPV="./datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/*/*.png"
LPV="./datasets/cityscapes/gtFine_trainvaltest/gtFine/val/*/*labelIds*.png"
echo "$IPTR"
python3 run.py -iw $IW -ih $IH -bs $BS -e $EP -nc $NC -iptr "$IPTR" -lptr "$LPTR" -ipv "$IPV" -lpv "$LPV"