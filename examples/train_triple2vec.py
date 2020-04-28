import sys
import argparse
from ray import tune
from beta_rec.train_engine import TrainEngine
from beta_rec.models.triple2vec import Triple2vecEngine
from beta_rec.utils.common_util import update_args
sys.path.append("../")


def parse_args():
    """
    Parse args from command line
    Returns:

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
    # these settings will be updated.
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
        "--percent",
        nargs="?",
        type=float,
        help="The percentage of the subset of the data set, only available on instacart data set.",
    )
    parser.add_argument(
        "--n_sample", nargs="?", type=int, help="Number of sampled triples."
    )
    parser.add_argument(
        "--temp_train",
        nargs="?",
        type=int,
        help="IF value >0, then the model will be trained based on the temporal feeding, else use normal trainning",
    )
    parser.add_argument(
        "--use_bias", nargs="?", type=int, help="",
    )
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Initialize learning rate.")
    parser.add_argument("--num_epoch", nargs="?", type=int, help="Number of max epoch.")

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
        self.sample_triple()
        self.engine = Triple2vecEngine(self.config)


def tune_train(config):
    """Train the model with a hypyer-parameter tuner (ray)

    Args:
        config (dict): All the parameters for the model

    Returns:

    """
    triple2vec = Triple2vec_train(config)
    best_performance = triple2vec.train()
    tune.track.log(best_ndcg=best_performance)
    triple2vec.test()


if __name__ == "__main__":
    args = parse_args()
    config = {}
    update_args(config, args)
    triple2vec = Triple2vec_train(config)
    triple2vec.train()
    triple2vec.test()
