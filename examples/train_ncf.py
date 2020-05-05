import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
from beta_rec.train_engine import TrainEngine
from beta_rec.models.gmf import GMFEngine
from beta_rec.models.mlp import MLPEngine
from beta_rec.models.ncf import NeuMFEngine
from beta_rec.datasets.nmf_data_utils import SampleGenerator
from beta_rec.utils.common_util import save_to_csv, update_args
from beta_rec.utils.monitor import Monitor

sys.path.append("../")


def parse_args():
    """ Parse args from command line

        Returns:
            args object.
    """
    parser = argparse.ArgumentParser(description="Run NCF..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/ncf_default.json",
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
        "--temp_train",
        nargs="?",
        type=int,
        help="IF value >0, then the model will be trained based on the temporal feeding, else use normal trainning",
    )
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
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


class NCF_train(TrainEngine):
    """ An instance class from the TrainEngine base class

        """

    def __init__(self, config):
        """Constructor

        Args:
            config (dict): All the parameters for the model
        """

        self.config = config
        super(NCF_train, self).__init__(self.config)
        self.load_dataset()
        self.sample_generator = SampleGenerator(ratings=self.dataset.train)
        # update model config
        common_config = self.config.copy()
        common_config.pop("gmf_config")
        common_config.pop("mlp_config")
        common_config.pop("neumf_config")
        self.config["gmf_config"].update(common_config)
        self.config["mlp_config"].update(common_config)
        self.config["neumf_config"].update(common_config)

    def train_epoch(self, engine, save_dir, temporal=False, time_step=0, t=0):
        epoch_bar = tqdm(range(self.config["max_epoch"]), file=sys.stdout)
        best_performance = 0
        for epoch in epoch_bar:
            print("Epoch {} starts !".format(epoch))
            print("-" * 80)
            if temporal:
                train_loader = self.sample_generator.instance_temporal_train_loader(
                    config["num_negative"],
                    config["batch_size"],
                    time_step=time_step,
                    t=t,
                )
            else:
                train_loader = self.sample_generator.instance_a_train_loader(
                    self.config["num_negative"], self.config["batch_size"]
                )
            engine.train_an_epoch(train_loader, epoch_id=epoch)
            """evaluate model on validation and test sets"""
            result = engine.evaluate(self.dataset.valid[0], epoch_id=epoch)
            test_result = engine.evaluate(self.dataset.test[0], epoch_id=epoch)
            engine.record_performance(result, test_result, epoch_id=epoch)
            if result["ndcg_at_k@10"] > best_performance:
                print(result)
                engine.save_checkpoint(model_dir=save_dir)
                best_performance = result["ndcg_at_k@10"]
                print("save model to" + self.gmf_save_dir)
                # best_result = result
        # save_result(result, result_file)

    def train(self):
        self.monitor = Monitor(
            log_dir=self.config["run_dir"], delay=1, gpu_id=self.gpu_id
        )

        # Train GMF
        self.gmf_engine = GMFEngine(self.config["gmf_config"])
        self.gmf_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["gmf_config"]["save_name"]
        )
        self.train_epoch(engine=self.gmf_engine, save_dir=self.gmf_save_dir)

        # Train MLP
        self.mlp_engine = MLPEngine(
            self.config["mlp_config"], gmf_config=self.config["gmf_config"]
        )
        self.mlp_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["mlp_config"]["save_name"]
        )
        self.train_epoch(engine=self.mlp_engine, save_dir=self.mlp_save_dir)

        # Train ncf
        self.neumf_engine = NeuMFEngine(
            self.config["neumf_config"],
            gmf_config=self.config["gmf_config"],
            mlp_config=self.config["mlp_config"],
        )
        self.neumf_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["neumf_config"]["save_name"]
        )
        self.train_epoch(engine=self.neumf_engine, save_dir=self.neumf_save_dir)

        self.config["run_time"] = self.monitor.stop()

    def temporal_train(self):
        self.monitor = Monitor(
            log_dir=self.config["run_dir"], delay=1, gpu_id=self.gpu_id
        )

        time_step = self.config["temp_train"]
        for t in range(time_step):
            self.gmf_engine = GMFEngine(self.config["gmf_config"])
            self.gmf_save_dir = (
                self.config["model_save_dir"] + self.config["gmf_config"]["save_name"]
            )
            self.train_epoch(
                engine=self.gmf_engine,
                save_dir=self.gmf_save_dir,
                temporal=True,
                time_step=time_step,
                t=t,
            )

            self.mlp_engine = MLPEngine(
                self.config["mlp_config"], gmf_config=self.config["gmf_config"]
            )
            self.mlp_save_dir = (
                self.config["model_save_dir"] + self.config["mlp_config"]["save_name"]
            )
            self.train_epoch(
                engine=self.mlp_engine,
                save_dir=self.mlp_save_dir,
                temporal=True,
                time_step=time_step,
                t=t,
            )

            self.neumf_engine = NeuMFEngine(
                self.config["neumf_config"],
                gmf_config=self.config["gmf_config"],
                mlp_config=self.config["mlp_config"],
            )
            self.neumf_save_dir = (
                self.config["model_save_dir"] + self.config["neumf_config"]["save_name"]
            )
            self.train_epoch(
                engine=self.neumf_engine,
                save_dir=self.neumf_save_dir,
                temporal=True,
                time_step=time_step,
                t=t,
            )

        self.config["run_time"] = self.monitor.stop()

    def test(self):
        """
        Prediction and evalution on test set
        """
        result_para = {
            "model": [self.config["model"]],
            "dataset": [self.config["dataset"]],
            "data_split": [self.config["data_split"]],
            "temp_train": [self.config["temp_train"]],
            "emb_dim": [int(self.config["emb_dim"])],
            "lr": [self.config["lr"]],
            "batch_size": [int(self.config["batch_size"])],
            "optimizer": [self.config["optimizer"]],
            "max_epoch": [self.config["max_epoch"]],
            "remarks": [self.config["model_run_id"]],
        }

        """
        load the best model in terms of the validate
        """
        self.neumf_engine.resume_checkpoint(model_dir=self.neumf_save_dir)
        for i in range(10):
            result = self.neumf_engine.evaluate(self.dataset.test[i], epoch_id=0)
            print(result)
            result["time"] = [self.config["run_time"]]
            result.update(result_para)
            result_df = pd.DataFrame(result)
            save_to_csv(result_df, self.config["result_file"])


if __name__ == "__main__":
    args = parse_args()
    config = {}
    update_args(config, args)
    ncf = NCF_train(config)
    ncf.train()
    ncf.test()
