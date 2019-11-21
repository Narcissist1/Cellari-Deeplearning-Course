# Cellari-Deeplearning-Course

Our network is based on Segnet. The task is to segment malicious cell from medical images.

The dataset can be found [here](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/)

reproduce the model

## 1 stage

200 epochs image size (128, 128) learning rate 1e-4 batch size 50

accuracy on test data 0.77

## 2 stage

200 epochs image size (192, 192) learning rate 5e-5 batch size 5

accuracy on test data 0.84

## 3 stage

200 epochs image size (256, 256) learning rate 5e-5 batch size 5

accuracy on test data 0.86

## 4 stage

200 epochs image size (256, 256) learning rate 1e-5 batch size 5

accuracy on test data 0.87

