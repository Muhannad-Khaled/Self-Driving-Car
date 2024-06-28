# Self-Driving-Car
Sure, I'll review the contents of the uploaded notebook and create a README file based on it. Let's start by examining the contents of the notebook.

The uploaded Jupyter notebook is in version 4.0 format and contains 27 cells. I will now extract key information from these cells to generate a comprehensive README file.

Based on the contents of the Jupyter notebook, here's a draft of a README file for the project:

---

# Self-Driving Car Project

This project implements a self-driving car model using deep learning techniques. The following sections provide an overview of the setup, data processing, model training, and evaluation steps involved in this project.

## Table of Contents
1. Introduction
2. Setup]
3. [Data Preprocessing]
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [Conclusion](#conclusion)

## Introduction
The goal of this project is to create a self-driving car model that can predict steering angles based on input images. The project leverages convolutional neural networks (CNNs) to achieve this.

## Setup
To get started, clone the project repository and install the necessary dependencies:

```python
import os
import numpy as np
import matplotlib.pyplot as plt

# Downloading data
!git clone https://github.com/Muhannad-Khaled/Self-Driving-Car
```

## Data Preprocessing
Data preprocessing involves organizing and preparing the images and steering data for training. Key steps include:

1. **Data Loading**:
    ```python
    def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail
    ```

2. **Data Distribution Analysis**:
    ```python
    num_bins = 25
    samples_per_bin = 400
    hist, bins = np.histogram(data['steering'], num_bins)
    ```

3. **Data Augmentation**:
    ```python
    def zoom(image):
        zoom = iaa.Affine(scale=(1, 1.3))
        image = zoom.augment_image(image)
    ```

4. **Image Preprocessing**:
    ```python
    def img_preprocess(img):
        img = img[60:135,:,:]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    ```

## Model Architecture
The model architecture is based on Nvidia's end-to-end deep learning model for self-driving cars:

```python
def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    ...
    return model
```

## Training
The model is trained using the processed images and steering angles:

```python
history = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                              steps_per_epoch=300,
                              epochs=10,
                              validation_data=batch_generator(X_valid, y_valid, 100, 0),
                              validation_steps=200,
                              verbose=1,
                              shuffle=1)
```

## Evaluation
Model evaluation is performed to assess the performance on the validation dataset:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
```

## Usage
To use the trained model for predictions:

```python
model.save('model.h5')
from google.colab import files
files.download('model.h5')
```

## Conclusion
This project demonstrates the application of deep learning techniques for self-driving car development. The implemented model can predict steering angles based on input images, showcasing the potential of autonomous driving technologies.

---

Feel free to adjust the content as needed for your specific use case or to add more detailed explanations where necessary.
