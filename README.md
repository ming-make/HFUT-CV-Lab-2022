# HFUT_CV_LAB_2022
Implementation for labs from HFUT Computer Vision course

There are three labs during this course, as shown below:
- lab1 : line detection based on Hough Transform
- lab2 : image segmentation based on any methods in CV
- lab3 : image classification based on any methods in CV

Description:

## lab1
Implement line detection algorithm based on hough transform.

```shell
python main.py
```

Test images in *assets* folder
Results in *results* folder

## lab2
Implement image segmentation algorithm based on meanshift

```shell
python meanshift.py
```

Test images in *assets* folder
Results in *results* folder

## lab3
Implement image identification algorithm based on CNN
- CNN(LeNet-5) for MNIST datasets
- modified LeNet-5 for CIFAR-10 datasets

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
