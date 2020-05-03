import pandas as pd
import torch
import numpy as np
from threading import Thread
from threading import Lock
import beta_rec.utils.evaluation as eval_model
from beta_rec.utils.constants import *
from beta_rec.utils.common_util import print_dict_as_table, save_to_csv, timeit
from tensorboardX import SummaryWriter
import socket
from prometheus_client import start_http_server, Gauge

lock_train_eval = Lock()
lock_test_eval = Lock()


def detect_port(port, ip="127.0.0.1"):
    """  Test whether the port is occupied.

    Args:
        port (int): port number
        ip (str): Ip address

    Returns:
        True -- it's possible to listen on this port for TCP/IPv4 or TCP/IPv6
                connections.
        False -- otherwise.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((ip, port))
        sock.listen(5)
        sock.close()
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.bind(("::1", port))
        sock.listen(5)
        sock.close()
    except socket.error as e:
        return False
        if rais:
            raise RuntimeError("The server is already running on port {0}".format(port))
    return True


def evaluate(data_df, predictions, metrics, k_li):
    """ Evaluate the performance of a prection by different metrics

    Args:
        data_df (DataFrame): the dataset to be evaluated
        predictions (narray): 1-D array. The predicted scores for each user-item pair in the dataset
        metrics (list):  metrics to be evaluated
        k_li (int or list): top k (s) to be evaluated

    Returns:
        result_dic (dict): Performance result

    """
    user_ids = data_df[DEFAULT_USER_COL].to_numpy()
    item_ids = data_df[DEFAULT_ITEM_COL].to_numpy()
    pred_df = pd.DataFrame(
        {
            DEFAULT_USER_COL: user_ids,
            DEFAULT_ITEM_COL: item_ids,
            DEFAULT_PREDICTION_COL: predictions,
        }
    )

    result_dic = {}
    if type(k_li) != list:
        k_li = [k_li]
    if 10 not in k_li:
        k_li.append(10)
    for k in k_li:
        for metric in metrics:
            eval_metric = getattr(eval_model, metric)
            result = eval_metric(data_df, pred_df, k=k)
            result_dic[metric + "@" + str(k)] = result
    return result_dic


@timeit
def train_eval_worker(
    testEngine, valid_df, test_df, valid_pred, test_pred, epoch, top_k=10
):
    """ Thread worker for the evaluation during training

    Args:
        testEngine:
        valid_df:
        test_df:
        valid_pred:
        test_pred:
        epoch:
        top_k:

    Returns:

    """
    valid_result = evaluate(valid_df, valid_pred, testEngine.metrics, top_k)
    test_result = evaluate(test_df, test_pred, testEngine.metrics, top_k)
    lock_train_eval.acquire()  # need to be test
    testEngine.record_performance(valid_result, test_result, epoch)
    testEngine.expose_performance(valid_result, test_result)
    if (
        valid_result[testEngine.config["validate_metric"]]
        > testEngine.best_valid_performance
    ):
        testEngine.n_no_update = 0
        testEngine.best_performance = valid_result[testEngine.config["validate_metric"]]
        print_dict_as_table(
            valid_result,
            tag=f"performance on validation at epoch {epoch}",
            columns=["metrics", "values"],
        )
    else:
        testEngine.n_no_update += 1
    lock_train_eval.release()
    # lock record and get best performance
    return valid_result, test_result


@timeit
def test_eval_worker(testEngine, eval_data_df, prediction, k_li=[5, 10, 20]):
    """
    Prediction and evaluation on the testing set
    """
    result_para = {
        "model": [testEngine.config["model"]],
        "dataset": [testEngine.config["dataset"]],
        "data_split": [testEngine.config["data_split"]],
        "emb_dim": [int(testEngine.config["emb_dim"])],
        "lr": [testEngine.config["lr"]],
        "batch_size": [int(testEngine.config["batch_size"])],
        "optimizer": [testEngine.config["optimizer"]],
        "max_epoch": [testEngine.config["max_epoch"]],
        "model_run_id": [testEngine.config["model_run_id"]],
        "run_time": [testEngine.config["run_time"]],
    }
    if "late_dim" in testEngine.config:
        result_para["late_dim"] = [int(testEngine.config["late_dim"])]
    if "alpha" in testEngine.config:
        result_para["alpha"] = [int(testEngine.config["alpha"])]
    if "activator" in testEngine.config:
        result_para["activator"] = [testEngine.config["activator"]]

    test_result_dic = evaluate(eval_data_df, prediction, testEngine.metrics, k_li)
    test_result_dic.update(result_para)
    lock_test_eval.acquire()  # need to be test
    result_df = pd.DataFrame(test_result_dic)
    save_to_csv(result_df, testEngine.config["result_file"])
    lock_test_eval.release()
    return test_result_dic


class EvalEngine(object):
    """The base evaluation engine.

    """

    def __init__(self, config):
        """ Constructor

        Args:
            config (dict): parameters for the model
        """
        self.config = config  # model configuration, should be a dic
        self.metrics = config["metrics"]
        self.validate_metric = config["validate_metric"]
        self.writer = SummaryWriter(log_dir=config["run_dir"])  # tensorboard writer
        self.writer.add_text(
            "model/config",
            pd.DataFrame(config.items(), columns=["parameters", "values"]).to_string(),
            0,
        )
        self.n_no_update = 0
        self.best_valid_performance = 0
        self.tunable = ["model", "dataset", "percent"]
        self.labels = (
            self.config["model"],
            self.config["dataset"],
            self.config["percent"],
        )
        self.init_prometheus_client()
        print("Initializing test engine ...")

    def predict(self, data_df, model):
        """ Make prediction for a trained model

        Args:
            data_df (DataFrame): A dataset to be evaluated
            model: A trained model

        Returns:
            array: predicted scores

        """
        user_ids = data_df[DEFAULT_USER_COL].to_numpy()
        item_ids = data_df[DEFAULT_ITEM_COL].to_numpy()
        predictions = np.array(
            model.predict(user_ids, item_ids)
            .flatten()
            .to(torch.device("cpu"))
            .detach()
            .numpy()
        )
        return predictions

    def train_eval(self, valid_data_df, test_data_df, model, epoch_id=0, k=10):
        """Evaluate the performance for a (validation) dataset with multiThread.

        Args:
            valid_data_df (DataFrame): A validation dataset
            test_data_df (DataFrame): A testing dataset
            model: trained model
            epoch_id: epoch_id
            k (int or list): top k result to be evaluate

        Returns:
            None

        """
        valid_pred = self.predict(valid_data_df, model)
        test_pred = self.predict(test_data_df, model)
        worker = Thread(
            target=train_eval_worker,
            args=(
                self,
                valid_data_df,
                test_data_df,
                valid_pred,
                test_pred,
                epoch_id,
                k,
            ),
        )
        worker.start()

    @timeit
    def test_eval(self, test_df_list, model, k=[5, 10, 20]):
        """Evaluate the performance for a (testing) dataset list with multiThread.

        Args:
            test_df_list (list): (testing) dataset list.
            model: trained model
            k (int or list): top k result to be evaluate

        Returns:
            None

        """

        if type(test_df_list) is not list:  # compatible for testing a single test set
            test_df_list = [test_df_list]

        for i, test_data_df in enumerate(test_df_list):
            test_pred = self.predict(test_data_df, model)
            worker = Thread(
                target=test_eval_worker,
                args=(self, test_data_df, test_pred, k),
                name="test_{}".format(i),
            )
            worker.start()

    def record_performance(self, valid_result, test_result, epoch_id):
        """Record perforance result on tensorboard

        Args:
            valid_result (dict): Performance result of validation set
            test_result (dict): Performance result of testing set
            epoch_id (int): epoch_id

        Returns:
            None

        """
        for metric in self.metrics:
            self.writer.add_scalars(
                "performance/" + metric,
                {
                    "valid": valid_result[metric + "@10"],
                    "test": test_result[metric + "@10"],
                },
                epoch_id,
            )

    def init_prometheus_client(self):
        """Initialize the prometheus http client

        Returns:
            None

        """
        if detect_port(8003):  # check if the port is available
            start_http_server(8003)
        gauges_test = {}
        gauges_valid = {}
        for metric in self.config["metrics"]:
            gauges_test[metric] = Gauge(
                metric + "_test",
                "Model Testing Performance under " + metric,
                self.tunable,
            )
            gauges_valid[metric] = Gauge(
                metric + "_valid",
                "Model Validation Performance under " + metric,
                self.tunable,
            )
        self.gauges_test = gauges_test
        self.gauges_valid = gauges_valid

    def expose_performance(self, valid_result, test_result):
        """
        Expose performance to a http_client
        Args:
            valid_result (dict): Performance result of validation set
            test_result (dict): Performance result of testing set

        Returns:
            None

        """
        for metric in self.config["metrics"]:
            self.gauges_valid[metric].labels(*self.labels).set(
                valid_result[metric + "@" + str(10)]
            )
            self.gauges_test[metric].labels(*self.labels).set(
                test_result[metric + "@" + str(10)]
            )
