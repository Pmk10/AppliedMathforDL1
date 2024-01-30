from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
from layered_modelsolution import get_mnist_data
import os

os.chdir('/Users/karthikeyanm/AppliedMathforDL/Assignment - 4')

def define_dense_model_single_layer(input_length, activation_f='sigmoid', output_length=10):
    """Define a dense model with a single layer for multiclass classification.
    input_length: the number of inputs
    activation_f: the activation function
    output_length: the number of outputs (number of neurons in the output layer)"""
    model = keras.Sequential([
        layers.Input(shape=(input_length,)),
        layers.Dense(output_length, activation=activation_f)
    ])
    return model

def define_dense_model_with_hidden_layer(input_length, 
                                         activation_func_array=['relu','sigmoid'],
                                         hidden_layer_size=10,
                                         output_length=10):
    """Define a dense model with a hidden layer for multiclass classification.
    input_length: the number of inputs
    activation_func_array: the activation function for the hidden layer and the output layer
    hidden_layer_size: the number of neurons in the hidden layer
    output_length: the number of outputs (number of neurons in the output layer)"""

    model = keras.Sequential([
        layers.Input(shape=(input_length,)),
        layers.Dense(hidden_layer_size, activation=activation_func_array[0]),
        layers.Dense(output_length, activation=activation_func_array[1])
    ])
    return model

def fit_mnist_model(x_train, y_train, model, epochs=2, batch_size=2):
    """Fit the model to the data for multiclass classification.
    compile the model and add parameters for the optimizer, the loss function, 
    and the metrics. Use categorical crossentropy for the loss function.

    then fit the model on the training data (pass the epochs and batch_size params)
    """
    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model
  
def evaluate_mnist_model(x_test, y_test, model):
    """Evaluate the model on the test data for multiclass classification.
    Hint: use model.evaluate() to evaluate the model on the test data.
    """
    # Convert labels to one-hot encoding
    y_test = keras.utils.to_categorical(y_test, 10)

    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy
