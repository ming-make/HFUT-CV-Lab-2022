# HFUT_CV_LAB_2022
Personal codes to labs from HFUT Computer Vision course

There are three labs during this course, as shown below:
- lab1 : line detection based on Hough Transform
- lab2 : image segmentation based on any methods in CV
- lab3 : image classification based on any methods in CV

Description:

lab1
- hough transform
- run main.py
- test images in assets
- result images in results

lab2
- meanshift
- run meanshift.py
- test images in assets
- result images in results

lab3
- CNN(LeNet-5) for MNIST datasets
- modified LeNet-5 for CIFAR-10 datasets
- train model based on MNIST datasets and output accuracy *$ python cnn.py mnist --option train*
- load pre-trained model, test and output accuracy *$ python cnn.py mnist --option test*
- train model based on CIFAR-10 datasets and output accuracy *$ python cnn.py cifar-10 --option train*
- load pre-trained model, test and output accuracy *$ python cnn.py cifar-10 --option test*
