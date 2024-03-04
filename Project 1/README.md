# Gradient descent

## Objectives

Project involves the development of a machine learning model to classify images from the CIFAR-10 dataset, focusing
on two specific classes: birds and planes. The model is based on a custom Multi-Layer Perceptron (MLP) implemented in
PyTorch. The process includes data loading, preprocessing, model training, and evaluation, including manual optimization
with momentum and weight decay adjustments.

## Libraries Used

- `torch`: For building and training the neural network model.
- `torchvision`: For loading and transforming the CIFAR-10 dataset.
- `matplotlib`: For plotting images and training statistics.
- `seaborn`: For generating confusion matrices.
- `sklearn`: For computing the classification report and confusion matrix.
- `numpy`: For data manipulation and analysis.
- `tabulate`: For formatting tables of results.
- `collections.Counter`: For counting label occurrences.
- `datetime`: For tracking and displaying time during training.

These libraries can be installed using the following command:

```bash
pip install torch torchvision matplotlib seaborn scikit-learn numpy tabulate
```

## How to Run

To run the project, execute the 'gradient_descent.ipynb' Jupyter notebook in sequence. The notebook includes detailed
steps from data loading and preprocessing to model training and evaluation. The code is accompanied by comments
explaining its purpose, making the process easy to follow.

## Key Components

1. **Data Loading and Preprocessing**: The CIFAR-10 dataset is loaded, analyzed, and preprocessed. It is split into
   training, validation, and test sets, focusing only on the bird and plane classes.
2. **Model Architecture**: A custom MLP (`MyMLP`) is defined with specific input and output dimensions and several
   hidden layers, using ReLU activation functions.
3. **Training**: Two training functions are provided: `train`, which uses PyTorch's SGD optimizer,
   and `train_manual_update`, which implements manual optimization with options for momentum and weight decay.
4. **Evaluation**: The best-performing model is identified based on validation accuracy and then evaluated on the test
   set to analyze its performance, including accuracy, loss, and confusion matrix.

## Results

The project explores various hyperparameter settings to optimize model performance. Through experimentation with
learning rate, momentum, and weight decay, the best model is selected and tested to provide understanding of its
capabilities in classifying images of birds and planes from the CIFAR-10 dataset.
