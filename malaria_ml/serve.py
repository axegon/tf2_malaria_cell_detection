import os
from glob import glob
import numpy as np
from tensorflow.keras.models import model_from_json
from .helpers import Helpers


class Serve(object):
    """Serves a pre-trained model.

    Parameters
    ----------
    model_path : str
        The path to a specific model or a list of models, in the second case, the latest will be picked."""

    _labels = {}

    def __init__(self, model_path):
        if not model_path.endswith('.h5') and os.path.isfile(model_path):
            model_path = sorted(glob(f"{model_path}/*.h5"))[::-1][0].replace(".h5", ".json")
        self._model = model_from_json(open(model_path.replace(".h5", ".json")).read())
        self._model.load_weights(f"{model_path}".replace(".json", ".h5"))

    def set_labels(self, labels):
        self._labels = labels

    def run(self, path, slices=None, resize=False, plot=False):
        """Runs the model on an image or a list of images.

        Parameters
        ----------
        path : str
            Path to specific image or a directory with images.
        slices : None|tuple
            Slice of the images, should you pass a directory.
        resize : False|tuple
            Resize the images, or leave them as they are.
        plot : bool
            Make plots of the images, only if the slice is less than 11.

        Returns
        -------
        np.array
            List of labeled predictions."""
        if os.path.isdir(path):
            images, paths = Helpers.reads_imdir(f"{path}", "png", slices, resize, True)
        else:
            images, paths = [Helpers.read_im(path, resize)], [path]
        images = np.array(images)
        results = self._model.predict(images)
        if plot and len(images) < 42:
            Helpers.show_images(images, 4, titles=[self._labels[int(r[0])] for r in results])
        return results
