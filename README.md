# How to use the code?

Original paper _"Predictive Performance of Deep Quantum Data Re-uploading Models"_

https://arxiv.org/abs/2505.20337

## Set up a Python environment

Choose Python version **3.10.x**

Run: **pip install -r requirements.txt**

## Generate binary datasets

- MNIST (digit 0/1, 12×12 pixels)

- CIFAR-10-Gray (airplane/automobile, grayscale, 12×12 pixels)

- CIFAR-10-RGB (airplane/automobile, RGB, 12×12 pixels)

## Run binary classification experiments

Run **../script/classification_mnist.sh** to perform the MNIST binary classification experiment, or integrate the configurations from the .sh file into **../real_world/classification.py** to run it directly.
