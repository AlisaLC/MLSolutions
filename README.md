# Solutions to ML practice problems
## HW2
### Linear Regression
I implemented linear regression using closed form solution and gradient descent from scratch with ridge, lasso and elastic net regularization. I also implemented cross validation to find the best hyperparameters for each model. I used the Boston Housing dataset from sklearn to test my models.

### Decision Tree
I implemented decision tree from scratch with entropy and information gain as the splitting criterion. I used breast cancer dataset to test my model.

### SVM
I performed PCA on olivetti faces dataset and used the first 100 principal components to train a SVM model. I used grid search to find the best hyperparameters for the model.

## HW3
### AdaBoost
I implemented AdaBoost from scratch and used it to classify the breast cancer dataset. The results were similar to sklearn's AdaBoostClassifier.

### Neural Network with Numpy
I implemented a deep neural network with `numpy`` and used it to classify CIFAR-10 dataset. Implemented modules include:
1. Linear Layer
1. ReLU Activation
1. Softmax Activation
1. Cross Entropy Loss
1. SGD Optimizer
1. Adam Optimizer
1. Batchnorm Layer
1. Dropout Layer

### Neural Network with PyTorch
I implemented a deep neural network with `PyTorch` and used it to classify EFIGI galaxy dataset.

## HW4
### CNN
I implemented a convolutional neural network with `PyTorch` and used it to classify MNIST dataset.

### Autoencoder
I implemented an autoencoder with `PyTorch` and used it to reconstruct Covid dataset. I used the encoder part of the autoencoder to extract features and plotted the features. The features were linearly separable much like PCA.

### LSTM
I implemented a LSTM network with `PyTorch` and used it to caption images from Flickr8k dataset. I used ResNet 50 to extract features from images and used the features as input to the LSTM network. The LSTM network was able to generate captions for images.