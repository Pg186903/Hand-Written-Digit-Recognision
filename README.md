## Project: Handwritten Digit Recognition with MNIST

### Overview

This project focuses on building a handwritten digit recognition system using the MNIST dataset, which contains 70,000 grayscale images (28x28 pixels) of digits (0–9). The workflow includes data preprocessing, visualization, model building with Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN), and evaluation.

# 

### Steps in the Notebook
#### 1. Environment Setup & Data Loading

* Libraries used:

  * tensorflow, keras → deep learning model building
  
  * numpy → data handling
  
  * matplotlib → visualization

        import tensorflow as tf
        from tensorflow import keras
        import matplotlib.pyplot as plt
        import numpy as np

* Checked dataset structure:

  * Training samples: len(X_train) → 60,000
  
  * Testing samples: len(X_test) → 10,000
  
  * Image shape: (28, 28)

#### 2. Data Exploration & Visualization

* Visualized digit samples using matshow().

* Confirmed labels matched the displayed digits.

      plt.matshow(X_train[0])
      print(y_train[0])  # Correct digit

#### 3. Preprocessing

* Normalized pixel values (0–255 → 0–1).

* Flattened images for ANN input (28x28 → 784).

      X_train = X_train / 255
      X_test = X_test / 255
      
      X_train_flattened = X_train.reshape(len(X_train), 28*28)
      X_test_flattened = X_test.reshape(len(X_test), 28*28)

#### 4. Model Building
* Baseline ANN without Hidden Layers

      model = keras.Sequential([
          keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
      ])

* ANN with Hidden Layers

  * Added dense hidden layers for better performance.

* CNN for Image Classification

  * Used Conv2D, MaxPooling2D, and Flatten layers.
  
  * CNN captured spatial features better than ANN.

#### 5. Training the Model

* Compiled with:

      model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])


* Trained for multiple epochs to improve accuracy.

      model.fit(X_train_flattened, y_train, epochs=5)

#### 6. Model Evaluation

* Evaluated on test data:

      test_loss, test_acc = model.evaluate(X_test_flattened, y_test)
      print("Test Accuracy:", test_acc)


* Visualized predictions vs actual labels.

* Compared ANN vs CNN performance.

# 

### Results

* ANN (no hidden layer): ~92% accuracy.

* ANN (with hidden layers): Improved accuracy (~95–96%).

* CNN: Achieved highest accuracy (~98%) on MNIST.

* CNN clearly outperformed ANN due to feature extraction from images.

# 

### Key Features of the Project

* Dataset: MNIST (10 categories, 28x28 grayscale images).

* Built and compared ANN vs CNN models.

* Applied preprocessing: normalization and flattening.

* Evaluation using accuracy and visualization of predictions.

* Extensions possible with:
  
  * Data augmentation (rotations, scaling, shifts).
  
  * Advanced CNN architectures (LeNet, ResNet).
  
  * Hyperparameter tuning (batch size, learning rate, optimizer).
