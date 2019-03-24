import argparse
from collections import Counter
from malaria_ml.serve import Serve


def main(FLAGS, _):
    model = Serve(FLAGS.model)
    model.set_labels({0: "Parasitized", 1: "Uninfected"})
    result = model.run(
        FLAGS.image,
        None if not FLAGS.slice else tuple(map(int, FLAGS.slice.split(','))),
        tuple(map(int, FLAGS.resize.split(","))),
        FLAGS.plot)
    if FLAGS.count_only:
        return Counter([int(i[0]) for i in result])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to a specific model or a directory with models(if directory, the newest one would be used."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Path to image or list of images in a directory"
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="Slice of the images in the directory, think arr[:X], where X = 100 for instance."
    )
    parser.add_argument(
        "--resize",
        type=str,
        default="141,141",
        help="Resizing the images to X,Y"
    )
    parser.add_argument(
        "--count_only",
        type=bool,
        default=False,
        help="Count The results only"
    )
    parser.add_argument(
        "--plot",
        type=bool,
        default=False,
        help="Plot the results, works only if the slice is less then or equal to 40."
    )
    FLAGS, unparsed = parser.parse_known_args()
    print(main(FLAGS, unparsed))
