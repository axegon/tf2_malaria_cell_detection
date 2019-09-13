"""Module used to train the model."""
# pylint: disable=import-error, too-many-arguments, too-many-instance-attributes
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, \
    Dropout, Dense, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split


class TrainModel:
    """The class handles the building and training of the model.

    Parameters
    ----------
    input_shape : np.array
        The shape of the input, that is width x height x channels.
    model_path : str
        Full or relative path to where the weights and model should be stored.
    logdir : str
        Path to the location where the Tensorboard logs would be saved."""
    _input_shape, _model = None, None
    _train_set, _train_labels = None, None
    _test_set, _test_labels = None, None
    _train_flow, _validation_flow = None, None
    _model_path = ""
    _accuracy, _loss = [], []

    def __init__(self, input_shape, model_path="", logdir="./logs"):
        self._model_path = model_path
        self._input_shape = input_shape
        self._build_model()
        self.tensorboard = TensorBoard(log_dir=logdir, update_freq='epoch')

    @property
    def model(self):
        """Access to self._model."""
        return self._model

    @property
    def loss(self):
        """Access to self._loss."""
        return self._loss

    @property
    def accuracy(self):
        """Access to self._accuracy."""
        return self._accuracy

    def _build_model(self):
        """Creates the model described bellow and compiles it.

        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        conv2d_1 (Conv2D)            (None, 139, 139, 32)      896
        _________________________________________________________________
        max_pooling2d_1 (MaxPooling2 (None, 69, 69, 32)        0
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 67, 67, 64)        18496
        _________________________________________________________________
        conv2d_3 (Conv2D)            (None, 65, 65, 64)        36928
        _________________________________________________________________
        max_pooling2d_2 (MaxPooling2 (None, 32, 32, 64)        0
        _________________________________________________________________
        conv2d_4 (Conv2D)            (None, 30, 30, 128)       73856
        _________________________________________________________________
        conv2d_5 (Conv2D)            (None, 28, 28, 128)       147584
        _________________________________________________________________
        conv2d_6 (Conv2D)            (None, 26, 26, 128)       147584
        _________________________________________________________________
        max_pooling2d_3 (MaxPooling2 (None, 13, 13, 128)       0
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 21632)             0
        _________________________________________________________________
        dropout_1 (Dropout)          (None, 21632)             0
        _________________________________________________________________
        dense_1 (Dense)              (None, 256)               5538048
        _________________________________________________________________
        dense_2 (Dense)              (None, 256)               65792
        _________________________________________________________________
        dense_3 (Dense)              (None, 1)                 257
        =================================================================
        Total params: 6,029,441
        Trainable params: 6,029,441
        Non-trainable params: 0
        _________________________________________________________________

        Returns
        -------
        None"""
        self._model = Sequential()
        self._model.add(Conv2D(32, (3, 3), activation="relu",
                               input_shape=self._input_shape))
        self._model.add(MaxPooling2D(2, 2))
        self._model.add(Conv2D(64, (3, 3), activation="relu"))
        self._model.add(Conv2D(64, (3, 3), activation="relu"))
        self._model.add(MaxPooling2D(2, 2))
        self._model.add(Conv2D(128, (3, 3), activation="relu"))
        self._model.add(Conv2D(128, (3, 3), activation="relu"))
        self._model.add(Conv2D(128, (3, 3), activation="relu"))
        self._model.add(MaxPooling2D(2, 2))
        self._model.add(Flatten())
        self._model.add(Dropout(0.3))
        self._model.add(Dense(256, activation="relu"))
        self._model.add(Dense(256, activation="relu"))
        self._model.add(Dense(1, activation="sigmoid"))
        self._model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(lr=1e-4),
            metrics=["acc"]
        )

    def summary(self, print_fn=print):
        """Prints the model summary.

        Parameters
        ----------
        print_fn : function
            Redirect the output to anything other than
            the built-in print function.

        Returns
        -------
        self"""
        self._model.summary(print_fn=print_fn)
        return self

    def feed_data(self, dataset, labels, test_size=0.3,
                  batch_size=150, random_state=42):
        """Creates an ImageDataGenerator from a given
        data set. The data is split into training and
        testing sets using scikit-learn's
        train_test_split function.

        Parameters
        ----------
        dataset : np.array
            List of rgb numpy representations of a set of images.
        labels : np.array
            Labels split accordingly to the dataset parameter.
        test_size : float
            What percentage of the dataset should be used for testing.
        batch_size : int
            Self-explanatory.
        random_state : int|None
            For reproducing identical test and train data sets.

        Returns
        -------
        None"""
        self._train_set, self._test_set, self._train_labels, self._test_labels = train_test_split(
            dataset,
            labels,
            test_size=test_size,
            random_state=random_state
        )
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=45,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.033,
            zoom_range=0.033,
            horizontal_flip=True,
            vertical_flip=True
        )

        self._train_flow = train_datagen.flow(
            self._train_set,
            self._train_labels,
            batch_size=batch_size
        )
        self._validation_flow = train_datagen.flow(
            self._test_set,
            self._test_labels,
            batch_size=batch_size
        )

    def train(self, epochs=42, steps_per_epoch=35, validation_steps=3):
        """Start the training process.

        Parameters
        ----------
        epochs : int
            Self-explanatory.
        steps_per_epoch : int
            Self-explanatory.
        validation_steps : int
            Self-explanatory.

        Returns
        -------
        None"""
        history = self._model.fit_generator(
            self._train_flow,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self._validation_flow,
            validation_steps=validation_steps,
            callbacks=[self.tensorboard]
        )
        self._accuracy = [*self._accuracy, *history.history["acc"]]
        self._loss = [*self._loss, *history.history["loss"]]
        self._save_model()

    def _save_model(self):
        """Stores the current state of the network.
        The method will save 2 files per completed.
        of epochs:
            model_<ISO_DATETIME>_<ACCURACY>_<LOSS>.json - The model definition.
            model_<ISO_DATETIME>_<ACCURACY>_<LOSS>.h5 - The model weights.

        Returns
        -------
        None"""
        accuracy, loss = self._accuracy[-1], self._loss[-1]
        model_name = f"model_{datetime.utcnow().isoformat()}_{accuracy}_{loss}"
        with open(f"{self._model_path}{model_name}.json", "w") as json_file:
            json_file.write(self._model.to_json())
        self._model.save_weights(f"{self._model_path}{model_name}.h5")

    def plot(self):
        """Simple plot of the training data at a given point.

        Returns
        -------
        None"""
        plt.plot(self._accuracy, label='Accuracy')
        plt.plot(self._loss, label='Loss')
        plt.title("Malaria cell detection")
        plt.legend(loc='center right')
        plt.show()
