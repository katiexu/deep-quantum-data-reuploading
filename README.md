# How to use the code?

Original paper _"Predictive Performance of Deep Quantum Data Re-uploading Models"_

https://arxiv.org/abs/2505.20337

## Set up the Python environment

Choose Python version **3.10.x**

Run **pip install -r requirements.txt**

## Run binary classification experiments

- MNIST (digit 0/1, 12×12 pixels)

Either run **../script/classification_mnist.sh** to perform the MNIST binary classification experiment, or integrate the configurations from the .sh file into **../real_world/classification.py** to run it directly.

<img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/921db61f-0daf-48e9-8531-c40ea5e8fa01" />

- CIFAR-10-Gray (airplane/automobile, grayscale, 12×12 pixels)

Either run **../script/classification_cifar10.sh** to perform the MNIST binary classification experiment, or integrate the configurations from the .sh file into **../real_world/classification.py** to run it directly.

- CIFAR-10-RGB (airplane/automobile, RGB, 12×12 pixels)

Either run **../script/classification_cifar10_rgb.sh** to perform the MNIST binary classification experiment, or integrate the configurations from the .sh file into **../real_world/classification.py** to run it directly.

## (Optional) Generate binary datasets

### The datasets for the binary classification experiments already exist in the "datasets" folder and can be used directly. If you need to generate new data, please follow the steps below.

<img width="544" height="365" alt="Screenshot from 2025-11-07 21-10-42" src="https://github.com/user-attachments/assets/8ea93a76-0c16-41d9-a690-f6067300aa35" />

- MNIST (digit 0/1, 12×12 pixels)

Run **../datasets_utils/mnist/generate_mnist_dataset.py** to generate the MNIST binary classification dataset. The data will be saved in the "datasets" folder.

- CIFAR-10-Gray (airplane/automobile, grayscale, 12×12 pixels)

Run **../datasets_utils/cifar_10_gray/generate_cifar10_gray_dataset.py** to generate the MNIST binary classification dataset. The data will be saved in the "datasets" folder.

- CIFAR-10-RGB (airplane/automobile, RGB, 12×12 pixels)

Run **../datasets_utils/cifar_10_rgb/generate_cifar10_rgb_dataset.py** to generate the MNIST binary classification dataset. The data will be saved in the "datasets" folder.
