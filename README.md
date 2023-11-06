# digit-recognition
# MNIST Digit Recognition using TensorFlow and Keras

This repository contains a Python script that uses TensorFlow and Keras to build and train a deep neural network for recognizing hand-written digits from the MNIST dataset.


## Project Overview
In this project, we use a deep neural network to classify hand-written digits from the MNIST dataset. The model is trained to predict the digit from 0 to 9 based on the pixel values of the input images.

## Prerequisites
- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib
- NumPy

You can install the required libraries using pip

3. Run the `mnist_digit_recognition.py` script to train and evaluate the model.

## Model Architecture
The neural network model consists of the following layers:
- Input Layer (Flatten layer)
- Dense Layer with 256 units and ReLU activation
- Dropout Layer with a dropout rate of 0.4
- Dense Layer with 128 units and ReLU activation
- Dropout Layer with a dropout rate of 0.4
- Output Layer with 10 units and softmax activation

## Training
The model is trained on the MNIST dataset with 20 epochs. Early stopping is employed with a patience of 5 to prevent overfitting. The best model weights are saved to a file named `mnist_model.h5`.

## Evaluation
The model's performance is evaluated on the test data, and the test accuracy is reported.

## Visualization
The script provides visualizations of the training and validation loss over epochs and displays sample test images with their true labels and predicted labels.

## Usage
You can use the `mnist_digit_recognition.py` script as a starting point for training and evaluating your own MNIST digit recognition models. You can modify the architecture, hyperparameters, and training settings according to your requirements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Feel free to contribute to or modify this code as needed. If you have any questions or suggestions, please create an issue or pull request.


