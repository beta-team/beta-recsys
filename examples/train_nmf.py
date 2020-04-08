import sys
sys.path.append("../")
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from beta_rec.models.gmf import GMFEngine
from beta_rec.models.mlp import MLPEngine
from beta_rec.models.neumf import NeuMFEngine
from beta_rec.datasets.nmf_data_utils import SampleGenerator
from beta_rec.utils.common_util import *
from beta_rec.utils.monitor import Monitor
from beta_rec.utils import data_util
from beta_rec.utils import logger
from beta_rec.datasets import dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run neumf..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/neumf_default.json",
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
        "--temp_train",
        nargs="?",
        type=int,
        help="IF value >0, then the model will be trained based on the temporal feeding, else use normal trainning",
    )
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
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


"""
update hyperparameters from command line
"""


def update_args(config, args):
    #     print(vars(args))
    for k, v in vars(args).items():
        if v != None:
            config[k] = v
            print("Received parameters form comand line:", k, v)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    config_file = args.config_file
    with open(config_file) as config_params:
        config = json.load(config_params)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    common_config = config["common_config"]
    root_dir = common_config["root_dir"]
    update_args(common_config, args)
    common_config["model"] = config["model"]
    common_config["model_str"] = (
        config["model"] + "_" + config["config_id"] + "_" + time_str
    )
    common_config["checkpoint_dir"] = (
        root_dir + common_config["checkpoint_dir"] + common_config["model_str"] + "/"
    )
    os.mkdir(common_config["checkpoint_dir"])
    gmf_config = config["gmf_config"]
    mlp_config = config["mlp_config"]
    neumf_config = config["neumf_config"]
    config.update(common_config)

    print(config)

    result_file = root_dir + common_config["result_dir"] + common_config["result_file"]
    log_file = root_dir + common_config["log_dir"] + common_config["model_str"]
    run_file = root_dir + common_config["run_dir"] + common_config["model_str"]

    """
    init logger
    """
    logger.init_std_logger(log_file)
    my_monitor = Monitor(log_dir=run_file)

    """
    Loading dataset
    """

    train_df, validate_df, test_df = dataset.load_split_dataset(config)

    print(len(train_df.index), len(validate_df[0].index), len(test_df[0].index))

    print(
        "train:",
        len(train_df.index),
        "validate:",
        len(validate_df[0].index),
        "test:",
        len(test_df[0].index),
    )

    data = data_util.Dataset(common_config)

    common_config["num_users"] = data.n_users
    common_config["num_items"] = data.n_items

    print(
        "num_users:",
        common_config["num_users"],
        " num_items:",
        common_config["num_items"],
    )
    gmf_config.update(common_config)
    mlp_config.update(common_config)
    neumf_config.update(common_config)
    config = common_config
    """
    log parameters
    """
    print("common_config:", common_config)
    print("gmf_config:", gmf_config)
    print("mlp_config:", mlp_config)
    print("neumf_config:", neumf_config)

    # DataLoader for training
    sample_generator = SampleGenerator(ratings=data.train)

    time_step = config["temp_train"]

    if time_step <= 1:
        # Train GMF
        engine = GMFEngine(gmf_config)
        model_save_dir = config["checkpoint_dir"] + gmf_config["save_name"]
        epoch_bar = tqdm(range(config["num_epoch"]), file=sys.stdout)
        best_performance = 0
        for epoch in epoch_bar:
            print("Epoch {} starts !".format(epoch))
            print("-" * 80)
            train_loader = sample_generator.instance_a_train_loader(
                config["num_negative"], config["batch_size"]
            )
            engine.train_an_epoch(train_loader, epoch_id=epoch)
            """evaluate model on vilidate and test sets"""
            result = engine.evaluate(data.validate[0], epoch_id=epoch)
            test_result = engine.evaluate(data.test[0], epoch_id=epoch)
            engine.record_performance(result, test_result, epoch_id=epoch)
            if result["ndcg_at_k@10"] > best_performance:
                print(result)
                engine.save_checkpoint(model_dir=model_save_dir)
                best_performance = result["ndcg_at_k@10"]
                best_result = result

        # save_result(result, result_file)

        # Train MLP
        mlp_save_dir = config["checkpoint_dir"] + mlp_config["save_name"]
        engine = MLPEngine(mlp_config, gmf_config=gmf_config)
        best_performance = 0
        for epoch in range(config["num_epoch"]):
            print("Epoch {} starts !".format(epoch))
            print("-" * 80)
            train_loader = sample_generator.instance_a_train_loader(
                config["num_negative"], config["batch_size"]
            )
            engine.train_an_epoch(train_loader, epoch_id=epoch)
            """evaluate model on vilidate and test sets"""
            result = engine.evaluate(data.validate[0], epoch_id=epoch)
            test_result = engine.evaluate(data.test[0], epoch_id=epoch)
            engine.record_performance(result, test_result, epoch_id=epoch)
            if result["ndcg_at_k@10"] > best_performance:
                print(result)
                engine.save_checkpoint(model_dir=mlp_save_dir)
                best_performance = result["ndcg_at_k@10"]
                best_result = result

        # save_result(result, result_file)

        # Train NeuMF

        neumf_save_dir = config["checkpoint_dir"] + neumf_config["save_name"]
        engine = NeuMFEngine(neumf_config, gmf_config=gmf_config, mlp_config=mlp_config)
        best_performance = 0
        for epoch in range(config["num_epoch"]):
            print("Epoch {} starts !".format(epoch))
            print("-" * 80)
            train_loader = sample_generator.instance_a_train_loader(
                config["num_negative"], config["batch_size"]
            )
            """evaluate model on vilidate and test sets"""
            result = engine.evaluate(data.validate[0], epoch_id=epoch)
            test_result = engine.evaluate(data.test[0], epoch_id=epoch)
            engine.record_performance(result, test_result, epoch_id=epoch)
            if result["ndcg_at_k@10"] > best_performance:
                print(result)
                engine.save_checkpoint(model_dir=neumf_save_dir)
                best_performance = result["ndcg_at_k@10"]
                best_result = result
    else:
        for t in range(time_step):
            print("=" * 80)
            print("train for time_step:", t)
            # Train GMF
            engine = GMFEngine(gmf_config)
            model_save_dir = config["checkpoint_dir"] + gmf_config["save_name"]
            epoch_bar = tqdm(range(config["num_epoch"]), file=sys.stdout)
            best_performance = 0
            for epoch in epoch_bar:
                print("Epoch {} starts !".format(epoch))
                print("-" * 80)
                train_loader = sample_generator.instance_temporal_train_loader(
                    config["num_negative"],
                    config["batch_size"],
                    time_step=time_step,
                    t=t,
                )
                engine.train_an_epoch(train_loader, epoch_id=epoch)
                """evaluate model on vilidate and test sets"""
                result = engine.evaluate(data.validate[0], epoch_id=epoch)
                test_result = engine.evaluate(data.test[0], epoch_id=epoch)
                engine.record_performance(result, test_result, epoch_id=epoch)
                if result["ndcg_at_k@10"] > best_performance:
                    print(result)
                    engine.save_checkpoint(model_dir=model_save_dir)
                    best_performance = result["ndcg_at_k@10"]
                    best_result = result

            # save_result(result, result_file)

            # Train MLP
            mlp_save_dir = config["checkpoint_dir"] + mlp_config["save_name"]
            engine = MLPEngine(mlp_config, gmf_config=gmf_config)
            best_performance = 0
            for epoch in range(config["num_epoch"]):
                print("Epoch {} starts !".format(epoch))
                print("-" * 80)
                train_loader = sample_generator.instance_temporal_train_loader(
                    config["num_negative"],
                    config["batch_size"],
                    time_step=time_step,
                    t=t,
                )
                engine.train_an_epoch(train_loader, epoch_id=epoch)
                """evaluate model on vilidate and test sets"""
                result = engine.evaluate(data.validate[0], epoch_id=epoch)
                test_result = engine.evaluate(data.test[0], epoch_id=epoch)
                engine.record_performance(result, test_result, epoch_id=epoch)
                if result["ndcg_at_k@10"] > best_performance:
                    print(result)
                    engine.save_checkpoint(model_dir=mlp_save_dir)
                    best_performance = result["ndcg_at_k@10"]
                    best_result = result

            # save_result(result, result_file)

            # Train NeuMF

            neumf_save_dir = config["checkpoint_dir"] + neumf_config["save_name"]
            engine = NeuMFEngine(
                neumf_config, gmf_config=gmf_config, mlp_config=mlp_config
            )
            best_performance = 0
            for epoch in range(config["num_epoch"]):
                print("Epoch {} starts !".format(epoch))
                print("-" * 80)
                train_loader = sample_generator.instance_temporal_train_loader(
                    config["num_negative"],
                    config["batch_size"],
                    time_step=time_step,
                    t=t,
                )
                """evaluate model on vilidate and test sets"""
                result = engine.evaluate(data.validate[0], epoch_id=epoch)
                test_result = engine.evaluate(data.test[0], epoch_id=epoch)
                engine.record_performance(result, test_result, epoch_id=epoch)
                if result["ndcg_at_k@10"] > best_performance:
                    print(result)
                    engine.save_checkpoint(model_dir=neumf_save_dir)
                    best_performance = result["ndcg_at_k@10"]
                    best_result = result
    run_time = my_monitor.stop()

    """
    Prediction and evalution on test set
    """
    result_para = {
        "model": [common_config["model"]],
        "dataset": [common_config["dataset"]],
        "data_split": [common_config["data_split"]],
        "temp_train": [common_config["temp_train"]],
        "emb_dim": [int(common_config["emb_dim"])],
        "lr": [common_config["lr"]],
        "batch_size": [int(common_config["batch_size"])],
        "optimizer": [common_config["optimizer"]],
        "num_epoch": [common_config["num_epoch"]],
        "remarks": [common_config["model_str"]],
    }

    """
    load the best model in terms of the validate
    """
    engine.resume_checkpoint(model_dir=neumf_save_dir)
    for i in range(10):
        result = engine.evaluate(data.test[i], epoch_id=0)
        print(result)
        result["time"] = [run_time]
        result.update(result_para)
        result_df = pd.DataFrame(result)
        save_result(result_df, result_file)
