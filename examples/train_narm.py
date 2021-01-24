"""isort:skip_file."""
import argparse
import os
import sys

sys.path.append("../")

from torch.utils.data import DataLoader
from tqdm import tqdm

from beta_rec.core.eval_engine import SeqEvalEngine
from beta_rec.core.train_engine import TrainEngine
from beta_rec.datasets.seq_data_utils import (
    SeqDataset,
    collate_fn,
    create_seq_db,
    dataset_to_seq_target_format,
    load_dataset,
    reindex_items,
)
from beta_rec.models.narm import NARMEngine
from beta_rec.utils.monitor import Monitor


def parse_args():
    """Parse args from command line.

    Returns:
        args object.
    """
    parser = argparse.ArgumentParser(description="Run NARM..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/narm_default.json",
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
    parser.add_argument("--root_dir", nargs="?", type=str, help="working directory")
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
        "--late_dim",
        nargs="?",
        type=int,
        help="Dimension of the latent layers.",
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Intial learning rate.")
    parser.add_argument("--num_epoch", nargs="?", type=int, help="Number of max epoch.")

    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    parser.add_argument("--optimizer", nargs="?", type=str, help="OPTI")
    parser.add_argument("--activator", nargs="?", type=str, help="activator")
    parser.add_argument("--alpha", nargs="?", type=float, help="ALPHA")

    return parser.parse_args()


class NARM_train(TrainEngine):
    """An instance class from the TrainEngine base class."""

    def __init__(self, config):
        """Initialize NARM_trian Class.

        Args:
            config (dict): All the parameters for the model.
        """
        self.config = config
        super(NARM_train, self).__init__(self.config)
        self.load_dataset_seq()
        self.build_data_loader()
        self.engine = NARMEngine(self.config)
        self.seq_eval_engine = SeqEvalEngine(self.config)

    def load_dataset_seq(self):
        """Build a dataset for model."""
        # ml = Movielens_100k()
        # ml.download()
        # ml.load_interaction()
        # self.dataset = ml.make_temporal_split(n_negative=0, n_test=0)

        ld_dataset = load_dataset(self.config)
        ld_dataset.download()
        ld_dataset.load_interaction()
        self.dataset = ld_dataset.make_temporal_split(n_negative=0, n_test=0)

        self.train_data = self.dataset[self.dataset.col_flag == "train"]
        self.valid_data = self.dataset[self.dataset.col_flag == "validate"]
        self.test_data = self.dataset[self.dataset.col_flag == "test"]

        # self.dataset = Dataset(self.config)
        self.config["dataset"]["n_users"] = self.train_data.col_user.nunique()
        self.config["dataset"]["n_items"] = self.train_data.col_item.nunique() + 1

    def build_data_loader(self):
        """Convert users' interactions to sequences.

        Returns:
            load_train_data (DataLoader): training set.
        """
        # reindex items from 1
        self.train_data, self.valid_data, self.test_data = reindex_items(
            self.train_data, self.valid_data, self.test_data
        )

        # data to sequences
        self.valid_data = create_seq_db(self.valid_data)
        self.test_data = create_seq_db(self.test_data)

        # convert interactions to sequences
        seq_train_data = create_seq_db(self.train_data)

        # convert sequences to (seq, target) format
        load_train_data = dataset_to_seq_target_format(seq_train_data)

        # define pytorch Dataset class for sequential datasets
        load_train_data = SeqDataset(load_train_data)

        # pad the sequences with 0
        self.load_train_data = DataLoader(
            load_train_data,
            batch_size=self.config["model"]["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
        )
        return self.load_train_data

    def _train(self, engine, train_loader, save_dir):
        """Train the model with epochs."""
        epoch_bar = tqdm(range(self.config["model"]["max_epoch"]), file=sys.stdout)
        for epoch in epoch_bar:
            print("Epoch {} starts !".format(epoch))
            print("-" * 80)
            if self.check_early_stop(engine, save_dir, epoch):
                break
            engine.train_an_epoch(train_loader, epoch=epoch)
            """evaluate model on validation and test sets"""

            # evaluation
            self.seq_eval_engine.train_eval_seq(
                self.valid_data, self.test_data, engine, epoch
            )

    def train(self):
        """Train and test NARM."""
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        train_loader = self.load_train_data
        self.engine = NARMEngine(self.config)
        self.narm_save_dir = os.path.join(
            self.config["system"]["model_save_dir"], self.config["model"]["save_name"]
        )
        self._train(self.engine, train_loader, self.narm_save_dir)
        self.config["run_time"] = self.monitor.stop()
        self.seq_eval_engine.test_eval_seq(self.test_data, self.engine)


if __name__ == "__main__":
    args = parse_args()
    narm = NARM_train(args)
    narm.train()
    # narm.test() have already implemented in train()
