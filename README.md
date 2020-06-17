![Flake8](https://github.com/mtszkw/crack-detection/workflows/Flake8/badge.svg)

# Surface Crack Classification

### Dataset

* [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2)
* [Surface Crack Detection Dataset | Kaggle](https://www.kaggle.com/arunrk7/surface-crack-detection)

The datasets contains images of various concrete surfaces with and without crack. The image data are divided into two as negative (without crack) and positive (with crack) in separate folder for image classification. Each class has 20000images with a total of 40000 images with 227 x 227 pixels with RGB channels. The dataset is generated from 458 high-resolution images (4032x3024 pixel) with the method proposed by Zhang et al (2016). High resolution images found out to have high variance in terms of surface finish and illumination condition. No data augmentation in terms of random rotation or flipping or tilting is applied.

##### Positive samples
![Positive samples](doc/positive-samples.PNG)

##### Negative samples
![Negative samples](doc/negative-samples.PNG)

### Approach

This code uses pretrained AlexNet model ([torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)) with trainable classifier part (see below layers in model.classifier). Dataset has been split into train, validation and test subsets. 

Model weights summary produced by Pytorch Lightning:

> 2  | model              | AlexNet           | 57 M
> 3  | model.features     | Sequential        | 2 M
> 4  | model.features.0   | Conv2d            | 23 K
> 5  | model.features.1   | ReLU              | 0
> 6  | model.features.2   | MaxPool2d         | 0
> 7  | model.features.3   | Conv2d            | 307 K
> 8  | model.features.4   | ReLU              | 0
> 9  | model.features.5   | MaxPool2d         | 0
> 10 | model.features.6   | Conv2d            | 663 K
> 11 | model.features.7   | ReLU              | 0
> 12 | model.features.8   | Conv2d            | 884 K
> 13 | model.features.9   | ReLU              | 0
> 14 | model.features.10  | Conv2d            | 590 K
> 15 | model.features.11  | ReLU              | 0
> 16 | model.features.12  | MaxPool2d         | 0
> 17 | model.avgpool      | AdaptiveAvgPool2d | 0
> 18 | model.classifier   | Sequential        | 54 M
> 19 | model.classifier.0 | Dropout           | 0
> 20 | model.classifier.1 | Linear            | 37 M
> 21 | model.classifier.2 | ReLU              | 0
> 22 | model.classifier.3 | Dropout           | 0
> 23 | model.classifier.4 | Linear            | 16 M
> 24 | model.classifier.5 | ReLU              | 0
> 25 | model.classifier.6 | Linear            | 4 K

### Results

