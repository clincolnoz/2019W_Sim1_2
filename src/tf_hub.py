# -*- coding: utf-8 -*-
"""TF Hub NN Model

"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import numpy as np
from datetime import datetime
import os


np.random.seed(1)


class TFHub:
    """Neural Network model implementation.
    Uses tf hub pretrained feature vector models
    """

    def __init__(self, kwargs):
        super().__init__()
        self.tf_url = kwargs["hub_layer_kwargs"]["tf_url"]
        self.trainable = kwargs["hub_layer_kwargs"]["trainable"]  # default False
        self.optional_hub_layer_kwargs = kwargs["hub_layer_kwargs"]["optional"]
        self.keras_layers = kwargs["lst_tf_keras_layers"]
        # self.optimizer = kwargs['optimizer']['optimizer']
        # self.loss = kwargs['optimizer']['loss']
        # self.metrics = kwargs['optimizer']['metrics']
        self.epochs = kwargs["model.fit"]["epochs"]
        self.train_generator = kwargs["train_generator"]
        self.development_generator = kwargs["development_generator"]
        # self.test_generator = kwargs["test_generator"]
        self.model_path = kwargs["model_path"]
        self.model_name = kwargs["model_name"]

    def setup_data(self, **kwargs):
        """Setup data function
        This function can be used by child classes to prepare data or perform
        other tasks that dont need to be repeated for every training run.
        Args:
            kwargs (:obj:`dict`): dictionary of optional arguments
        """
        pass

    def setup_model(self):
        """retrieve tensorflow hub model for use in setup_model
        Args:
            kwargs (:obj:`dict`): dictionary of optional arguments
            kwargs['trainable'] (bool): True to unfreeze weights, default False
            kwargs['arguments'] (dict): optionally, a dict with additional keyword arguments passed to the callable. 
                                        These must be JSON-serializable to save the Keras config of this layer.
                                        eg dict(batch_norm_momentum=0.997)
            **kwargs: 'output_shape': A tuple with the (possibly partial) output shape of the callable without 
                      leading batch size. Other arguments are pass into the Layer constructor.
        """
        hub_layer = hub.KerasLayer(
            self.tf_url, trainable=self.trainable, **self.optional_hub_layer_kwargs
        )

        self.model = tf.keras.Sequential(
            [tf.keras.Input(shape=[224, 224, 3]), hub_layer]
        )
        # self.model.add(self.keras_layers)
        # self.model.add(tf.keras.layers.Dropout(rate=0.2))
        self.model.add(
            tf.keras.layers.Dense(
                self.train_generator.num_classes, activation="softmax",
            )
        )
        # self.model.build([None, 224, 224, 3])  # Batch input shape.

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

        # self.model.compile(
        #     optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
        #     loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        #     metrics=["accuracy"],
        # )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        # class CollectBatchStats(tf.keras.callbacks.Callback):
        #     def __init__(self):
        #         self.batch_losses = []
        #         self.batch_precision = []
        #         self.batch_recall = []

        #     def on_train_batch_end(self, batch, logs=None):
        #         self.batch_losses.append(logs['loss'])
        #         self.batch_precision.append(logs[tf.keras.metrics.Precision().result()])
        #         self.batch_recall.append(logs[tf.keras.metrics.Recall().result()])
        #         self.model.reset_metrics()

        # batch_stats_callback = CollectBatchStats()

        # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # if not os.path.exists(logdir):
        #     parent = os.path.split(logdir)[0]
        #     if not os.path.exists(parent):
        #         parent2 = os.path.split(parent)[0]
        #         if not os.path.exists(parent2):
        #             os.mkdir(parent2)
        #         os.mkdir(parent)
        #     os.mkdir(logdir)
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        path = "{}{}/bestmodel".format(self.model_path, self.model_name)
        if not os.path.exists(path):
            parent = os.path.split(path)[0]
            if not os.path.exists(parent):
                parent2 = os.path.split(parent)[0]
                if not os.path.exists(parent2):
                    os.mkdir(parent2)
                os.mkdir(parent)
            os.mkdir(path)

        path = path + "/" + self.model_name

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=path + "_{epoch}.h5",
                # Path where to save the model
                # The two parameters below mean that we will overwrite
                # the current checkpoint if and only if
                # the `val_loss` score has improved.
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
            # batch_stats_callback
            # tensorboard_callback,
        ]

        steps_per_epoch = (
            self.train_generator.samples // self.train_generator.batch_size
        )
        validation_steps = (
            self.development_generator.samples // self.development_generator.batch_size
        )
        print([steps_per_epoch, validation_steps])
        hist = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.development_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
        ).history
        return hist

    def predict(self, x, steps, **kwargs):
        """Model predict function.
        Model scoring.
        This method is final. Signature will be checked at runtime!
        Args:
            kwargs (:obj:`dict`): dictionary of optional arguments.
        Returns:
            yhat: numerical matrix containing the predicted responses.
        """
        return self.model.predict(x, steps=steps)

    def evaluate(self, xy_gen, steps, **kwargs):
        """Model predict and evluate.
        This method is final. Signature will be checked at runtime!
        Args:
            kwargs (:obj:`dict`): dictionary of optional arguments.
        Returns:
            metrics: to be defined!
        """

        evaluation = self.model.evaluate(xy_gen, steps=steps)
        return evaluation

    def save(self, name, version):
        """Saves the model.
        Save the model in binary format on local storage.
        This method is final. Signature will be checked at runtime!
        Args:
            name (str): name for the model to use for saving
            version (str): version of the model to use for saving
        """
        self.model.save(
            filepath, include_optimizer=True,
        )

    def load(self, filepath):
        """Loads the model.
        Load the model from local storage.
        This method is final. Signature will be checked at runtime!
        Args:
            name (str): name of the model to load
            version (str): version of the model to load
        """
        self.model = tf.keras.models.load_model(
            filepath, compile=True, custom_objects={"KerasLayer": hub.KerasLayer},
        )
        for i, layer in enumerate(self.model.layers):
            if layer.name == "keras_layer":
                self.model.layers[i].trainable = self.trainable
        # self.model.build([None, 224, 224, 3])  # Batch input shape.
