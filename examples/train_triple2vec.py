import argparse

from ray import tune

from beta_rec.core.train_engine import TrainEngine
from beta_rec.models.triple2vec import Triple2vecEngine
from beta_rec.utils.common_util import DictToObject, str2bool


def parse_args():
    """ Parse args from command line

        Returns:
            args object.
    """
    parser = argparse.ArgumentParser(description="Run Triple2vec..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/triple2vec_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
    )
    # If the following settings are specified with command line,
    # These settings will used to update the parameters received from the config file.
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        help="Options are: tafeng, dunnhunmby and instacart",
    )
    parser.add_argument(
        "--data_split",
        nargs="?",
        type=str,
        help="Options are: leave_one_out and temporal",
    )
    parser.add_argument(
        "--root_dir", nargs="?", type=str, help="working directory",
    )
    parser.add_argument(
        "--tune", nargs="?", type=str2bool, help="Tun parameter",
    )
    parser.add_argument(
        "--n_sample", nargs="?", type=int, help="Number of sampled triples."
    )
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Initialize learning rate.")
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")
    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    parser.add_argument("--optimizer", nargs="?", type=str, help="OPTI")
    return parser.parse_args()


class Triple2vec_train(TrainEngine):
    """ An instance class from the TrainEngine base class

    """

    def __init__(self, config):
        """Constructor

        Args:
            config (dict): All the parameters for the model
        """

        self.config = config
        super(Triple2vec_train, self).__init__(self.config)
        self.load_dataset()
        self.train_loader = self.data.sample_triple()
        self.engine = Triple2vecEngine(self.config)


def tune_train(config):
    """Train the model with a hypyer-parameter tuner (ray)

    Args:
        config (dict): All the parameters for the model

    Returns:

    """
    train_engine = Triple2vec_train(DictToObject(config))
    best_performance = train_engine.train()
    tune.track.log(valid_metric=best_performance)
    train_engine.test()


if __name__ == "__main__":
    args = parse_args()
    train_engine = Triple2vec_train(args)
    if args.tune:
        train_engine.tune(tune_train)
    else:
        train_engine.train()
        train_engine.test()
