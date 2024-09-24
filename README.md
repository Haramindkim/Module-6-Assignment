# Fashion MNIST Classification

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. 

## Requirements
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

## Installation
1. Ensure you have Python 3.7 or later installed.

### Importing Libraries
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
```
This section imports the necessary libraries: TensorFlow for the neural network, NumPy for numerical operations, and Matplotlib for plotting.

```python
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
```

This code loads the Fashion MNIST dataset, normalizes the pixel values to be between 0 and 1, and reshapes the images to include a channel dimension.

```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```
This creates a sequential model with three convolutional layers, two max pooling layers, and two dense layers. The final layer has 10 neurons, one for each class in the dataset.

```python
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_split=0.2)
```
This code trains the model on the training data for 10 epochs, using 20% of the training data for validation.

```python
predictions = model.predict(np.array([image1, image2]))
```
This uses the trained model to make predictions on two sample images.
