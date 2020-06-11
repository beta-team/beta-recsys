import argparse
import math
import os
import sys

sys.path.append("../")

from beta_rec.models.vbcar import VBCAREngine
from beta_rec.train_engine import TrainEngine
from beta_rec.utils.common_util import update_args
from beta_rec.utils.constants import MAX_N_UPDATE
from beta_rec.utils.monitor import Monitor

from ray import tune

from tqdm import tqdm


def parse_args():
    """ Parse args from command line

        Returns:
            args object.
    """
    parser = argparse.ArgumentParser(description="Run VBCAR..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/vbcar_default.json",
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
    parser.add_argument("--root_dir", nargs="?", type=str, help="working directory")
    parser.add_argument(
        "--percent",
        nargs="?",
        type=float,
        help="The percentage of the subset of the dataset, only availbe on instacart dataset.",
    )
    parser.add_argument(
        "--n_sample", nargs="?", type=int, help="Number of sampled triples."
    )
    parser.add_argument("--sub_set", nargs="?", type=int, help="Subset of dataset.")
    parser.add_argument(
        "--temp_train",
        nargs="?",
        type=int,
        help="IF value >0, then the model will be trained based on the temporal feeding, else use normal trainning.",
    )
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument(
        "--late_dim", nargs="?", type=int, help="Dimension of the latent layers.",
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Intial learning rate.")
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")
    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    parser.add_argument("--optimizer", nargs="?", type=str, help="OPTI")
    parser.add_argument("--activator", nargs="?", type=str, help="activator")
    parser.add_argument("--alpha", nargs="?", type=float, help="ALPHA")
    return parser.parse_args()


class VBCAR_train(TrainEngine):
    """ An instance class from the TrainEngine base class

    """

    def __init__(self, config):
        """Constructor

                Args:
                    config (dict): All the parameters for the model
        """
        self.config = config
        super(VBCAR_train, self).__init__(self.config)
        self.load_dataset()
        self.train_data = self.dataset.sample_triple()
        self.config["alpha_step"] = (1 - self.config["alpha"]) / (
            self.config["max_epoch"]
        )
        self.engine = VBCAREngine(self.config)

    def train(self):
        """Default train implementation

        """
        assert hasattr(self, "engine"), "Please specify the exact model engine !"
        self.monitor = Monitor(
            log_dir=self.config["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        self.engine.data = self.dataset
        print("Start training... ")
        epoch_bar = tqdm(range(self.config["max_epoch"]), file=sys.stdout)
        for epoch in epoch_bar:
            print(f"Epoch {epoch} starts !")
            print("-" * 80)
            if epoch > 0 and self.eval_engine.n_no_update == 0:
                # previous epoch have already obtained better result
                self.engine.save_checkpoint(
                    model_dir=os.path.join(self.config["model_save_dir"], "model.cpk")
                )

            if self.eval_engine.n_no_update >= MAX_N_UPDATE:
                print(
                    "Early stop criterion triggered, no performance update for {:} times".format(
                        MAX_N_UPDATE
                    )
                )
                break
            data_loader = self.build_data_loader()
            self.engine.train_an_epoch(data_loader, epoch_id=epoch)
            self.eval_engine.train_eval(
                self.dataset.valid[0], self.dataset.test[0], self.engine.model, epoch
            )
            # anneal alpha
            self.engine.model.alpha = min(
                self.config["alpha"] + math.exp(epoch - self.config["max_epoch"] + 20),
                1,
            )
            """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
            lr = self.config["lr"] * (0.5 ** (epoch // 10))
            for param_group in self.engine.optimizer.param_groups:
                param_group["lr"] = lr
        self.config["run_time"] = self.monitor.stop()
        return self.eval_engine.best_valid_performance


def tune_train(config):
    VBCAR = VBCAR_train(config)
    best_performance = VBCAR.train()
    tune.track.log(best_ndcg=best_performance)
    VBCAR.test()


if __name__ == "__main__":
    args = parse_args()
    config = {}
    update_args(config, args)
    VBCAR = VBCAR_train(config)
    VBCAR.train()
    VBCAR.test()
