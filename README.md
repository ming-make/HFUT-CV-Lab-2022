# HFUT CV LAB 2022
Implementation for labs from HFUT Computer Vision course

There are three labs during this course, as shown below:
- Lab1 : Line detection based on Hough Transform
- Lab2 : Image segmentation based on any methods in CV
- Lab3 : Image classification based on any methods in CV

Description:

## Platform
- Windows 10
- Pycharm 2022.1
- Python 3.8

## lab1
Implement line detection algorithm based on hough transform.

These libraries are needed:
- opencv-python 4.5.5.64
- numpy 1.22.3

```shell
python main.py
```

Test images in *assets* folder
Results in *results* folder

## lab2
Implement image segmentation algorithm based on meanshift

These libraries are needed:
- opencv-python 4.5.5.64
- numpy 1.22.3
- scipy 1.4.1

```shell
python meanshift.py
```

Test images in *assets* folder
Results in *results* folder

## lab3
Implement image identification algorithm based on CNN
- CNN(LeNet-5) for MNIST datasets
- modified LeNet-5 for CIFAR-10 datasets

These libraries are needed:
- matplotlib 3.5.1
- numpy 1.22.3
- sklearn 0.0
- tensorflow 2.10.0
- keras 2.10.0

Train model based on MNIST datasets and output accuracy

```shell
python cnn.py mnist --option train
```

Load pre-trained model, test and output accuracy

```shell
python cnn.py mnist --option test
```

Train model based on CIFAR-10 datasets and output accuracy

```shell
python cnn.py cifar-10 --option train
```

Load pre-trained model, test and output accuracy

```shell
python cnn.py cifar-10 --option test
```
