#!/bin/bash

if [ -d "./cell_images" ]; then
    echo "It appears you have already downloaded the data."
    echo "Remove ./cell_images if you want to download it again"
else
    rm -rf cell_images*
    wget https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip
    unzip --qq cell_images.zip
fi