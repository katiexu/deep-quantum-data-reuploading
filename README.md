# How to use the code?

Original paper _"Predictive Performance of Deep Quantum Data Re-uploading Models"_

https://arxiv.org/abs/2505.20337

<img width="2000" height="1000" alt="image" src="https://github.com/user-attachments/assets/fcfd945f-a45b-4f8d-8b1d-fe662b6f4ce3" />

## Set up the Python environment

Choose Python version **3.10.x**

Run **pip install -r requirements.txt**

## Run binary classification experiments

- MNIST (digit 0/1, 12×12 pixels)

Either run **../script/classification_mnist.sh** to perform the MNIST binary classification experiment, or integrate the configurations from the .sh file into **../real_world/classification.py** to run it directly.

<img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/f4878318-b50c-4f37-9859-ee54f4e1dc95" />

- CIFAR-10-Gray (airplane/automobile, grayscale, 12×12 pixels)

Either run **../script/classification_cifar10.sh** to perform the CIFAR-10 (Gray) binary classification experiment, or integrate the configurations from the .sh file into **../real_world/classification.py** to run it directly.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/83312929-1e86-437a-8420-7fcbef4a71ab" />

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/1bc17cfd-5c9d-4394-b92f-a85e563873cd" />

- CIFAR-10-RGB (airplane/automobile, RGB, 12×12 pixels)

Either run **../script/classification_cifar10_rgb.sh** to perform the CIFAR-10 (RGB) binary classification experiment, or integrate the configurations from the .sh file into **../real_world/classification.py** to run it directly.

## (Optional) Generate binary datasets

### The datasets for the binary classification experiments already exist in the "datasets" folder and can be used directly. If you need to generate new data, please follow the steps below.

<img width="400" height="300" alt="Screenshot from 2025-11-07 21-10-42" src="https://github.com/user-attachments/assets/8ea93a76-0c16-41d9-a690-f6067300aa35" />

- MNIST (digit 0/1, 12×12 pixels)

Run **../datasets_utils/mnist/generate_mnist_dataset.py** to generate the MNIST binary classification dataset. The data will be saved in the "datasets" folder.

- CIFAR-10-Gray (airplane/automobile, grayscale, 12×12 pixels)

Run **../datasets_utils/cifar_10_gray/generate_cifar10_gray_dataset.py** to generate the CIFAR-10 (Gray) binary classification dataset. The data will be saved in the "datasets" folder.

- CIFAR-10-RGB (airplane/automobile, RGB, 12×12 pixels)

Run **../datasets_utils/cifar_10_rgb/generate_cifar10_rgb_dataset.py** to generate the CIFAR-10 (RGB) binary classification dataset. The data will be saved in the "datasets" folder.
