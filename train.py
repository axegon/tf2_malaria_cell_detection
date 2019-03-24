import argparse
import yaml
import numpy as np
from malaria_ml.train import TrainModel
from malaria_ml.helpers import Helpers


def main(FLAGS):
    input_shape = FLAGS["image_size"]
    input_shape.append(3)
    model = TrainModel(input_shape=input_shape, model_path=FLAGS["output_models"])
    model.summary()
    dataset = np.concatenate(
        np.array(
            [Helpers.reads_imdir(f"{FLAGS['train_images']}{imtype}", "png", (None, FLAGS["train_size"]), (141, 141))
             for imtype in FLAGS["labels"]]
        ),
        axis=0
    )
    labels = np.concatenate((np.zeros(FLAGS["train_size"]), np.ones(FLAGS["train_size"])), axis=0)
    model.feed_data(dataset, labels, FLAGS["test_size"], FLAGS["batch_size"], FLAGS["random_state"])
    model.train(epochs=FLAGS["epochs"], steps_per_epoch=FLAGS["steps_per_epoch"],
                validation_steps=FLAGS["validation_steps"])
    if FLAGS["plot_training_data"]:
        model.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration YAML file."
    )
    config, _ = parser.parse_known_args()
    with open(config.config) as f:
        FLAGS = yaml.load(f)
    main(FLAGS)
