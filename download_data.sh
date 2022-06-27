#!/bin/bash

wget -N http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz -O ./datasets/maps.tar.gz
mkdir -p ./datasets/maps

tar -zxvf ./datasets/maps.tar.gz -C ./datasets/
rm ./datasets/maps.tar.gz