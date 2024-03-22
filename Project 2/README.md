# Object Localization & Object Detection

## Objectives

This project involves the development of machine learning models for two specific tasks: object localization and object
detection, using an augmented version of the MNIST dataset. The dataset features randomly placed, rotated, and resized
digits against noisy backgrounds, in images of dimensions 48x60. The main goals of the project include:

1. **Understanding Convolutional Neural Networks (CNNs):** Exploring the mechanics and applications of CNNs for
   image-based tasks.
2. **Mastering Object Localization:** Learning to classify images and pinpoint the locations of objects within them,
   assuming a single object instance per image.
3. **Exploring Object Detection:** Expanding the capabilities of object localization to accommodate multiple objects
   within a single image, increasing the complexity and applicability of the models.

## Libraries Used

- **`PyTorch` & `torchvision`:** For constructing, training, and evaluating neural network models, and managing the MNIST
  dataset.
- **`matplotlib` & `seaborn`:** For visualizations, including plotting images, training statistics, and confusion matrices.
- **`scikit-learn`:** For generating detailed classification reports and confusion matrices.
- **`numpy`:** For data manipulation and mathematical operations.
- **`collections.Counter`:** For analyzing the frequency of labels within the dataset.
- **`datetime`:** For managing time-related information during model training.

To install the necessary libraries:

```bash
pip install torch torchvision matplotlib seaborn scikit-learn numpy 
```

## How to Run

To run the project, execute the 'main.ipynb' Jupyter notebook in sequence. The code is accompanied by comments
explaining its purpose, making the process easy to follow.

## Results

Through a process model development, the project aims to achieve optimal performance for both object localization and
detection tasks. It underscores the effectiveness of CNNs in handling image processing tasks and provides insights into
model architecture design, adaptation of loss functions, and evaluation techniques.
