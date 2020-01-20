# -*- coding: utf-8 -*-
"""Audio Features Model
Basic tensorflow neural network implementation using Keras.

"""

import tensorflow as tf
from tensorflow import keras


class AudioFeatures():
    """Neural Network model implementation.
    SDK users can use this class to create and train Keras models or
    subclass this class to define custom neural networks.
    """
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.1
        self.epochs = 50
        self.num_microbatches = 250
        self.verbose = 0
        self.metrics = ['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]
        self.model = None
        
    def setup_data(self, **kwargs):
        """Setup data function
        This function can be used by child classes to prepare data or perform
        other tasks that dont need to be repeated for every training run.
        Args:
            kwargs (:obj:`dict`): dictionary of optional arguments
        """
        pass

    def setup_model(self, **kwargs):
        """Setup model function
        Implementing child classes can use this method to define the
        Keras model.
        Args:
            kwargs (:obj:`dict`): dictionary of optional arguments
        """
        self.model = keras.Sequential([
            keras.layers.Input(kwargs['n_features']),
            keras.layers.Dense(40, activation='relu'),
            keras.layers.Dense(40, activation='relu'),
            keras.layers.Dense(40, activation='relu'),
            keras.layers.Dense(40, activation='relu'),
            keras.layers.Dense(kwargs['n_labels'], activation='softmax')]
        )

    def prepare(self, **kwargs):
        """called before model fit on every run.
        Implementing child classes can use this method to prepare
        data for model training (preprocess data).
        Args:
            kwargs (:obj:`dict`): dictionary of optional arguments
        """
        pass

    def fit(self, **kwargs):
        """Model fit function.
        This method is final. Signature will be checked at runtime!
        Args:
            kwargs (:obj:`dict`): dictionary of optional arguments.
                preprocessed data, feature columns
        """
        X_train = kwargs['X_train']
        y_train = kwargs['y_train']

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy()
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=self.metrics)
        hist = self.model.fit(X_train,
                       y_train,
                       epochs=self.epochs).history
        return hist

    def predict(self, **kwargs):
        """Model predict function.
        Model scoring.
        This method is final. Signature will be checked at runtime!
        Args:
            kwargs (:obj:`dict`): dictionary of optional arguments.
        Returns:
            yhat: numerical matrix containing the predicted responses.
        """
        X_test = kwargs['X_test']
        
        return self.model.predict(X_test)

    def evaluate(self, **kwargs):
        """Model predict and evluate.
        This method is final. Signature will be checked at runtime!
        Args:
            kwargs (:obj:`dict`): dictionary of optional arguments.
        Returns:
            metrics: to be defined!
        """
        X_test = kwargs['X_test']
        y_test = kwargs['y_test']

        evaluation = self.model.evaluate(X_test,y_test)
        return evaluation
    
    def save(self, filepath):
        """Saves the model.
        Save the model in binary format on local storage.
        This method is final. Signature will be checked at runtime!
        Args:
            name (str): name for the model to use for saving
            version (str): version of the model to use for saving
        """
        self.model.save(filepath,
                        include_optimizer=True)

    def load(self, filepath):
        """Loads the model.
        Load the model from local storage.
        This method is final. Signature will be checked at runtime!
        Args:
            name (str): name of the model to load
            version (str): version of the model to load
        """
        self.model = tf.keras.models.load_model(
            filepath, compile=True)