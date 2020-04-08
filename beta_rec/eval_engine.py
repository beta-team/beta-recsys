from threading import Thread
from threading import Lock
import beta_rec.utils.evaluation as eval_model
from beta_rec.utils.constants import *
from beta_rec.utils.common_util import *
from tensorboardX import SummaryWriter
import socket
from prometheus_client import start_http_server, Gauge

lock_train_eval = Lock()
lock_test_eval = Lock()


def dict2str(dic):
    dic_str = (
        "Configs: \n"
        + "\n".join([str(k) + ":\t" + str(v) for k, v in dic.items()])
        + "\n"
    )
    print(dic_str)
    return dic_str


def detect_port(port, ip="127.0.0.1"):
    """
    test if the port is occupied.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        print("{0} is opened".format(port))
        return True
    except:
        print("{0} is closed".format(port))
        return False


@timeit
def evaluate(data_df, predictions, metrics, TOP_K):
    user_ids = data_df[DEFAULT_USER_COL].to_numpy()
    item_ids = data_df[DEFAULT_ITEM_COL].to_numpy()
    ratings = data_df[DEFAULT_RATING_COL].to_numpy()
    pred_df = pd.DataFrame(
        {
            DEFAULT_USER_COL: user_ids,
            DEFAULT_ITEM_COL: item_ids,
            DEFAULT_PREDICTION_COL: predictions,
        }
    )

    result_dic = {}
    if type(TOP_K) != list:
        TOP_K = [TOP_K]
    if 10 not in TOP_K:
        TOP_K.append(10)
    for k in TOP_K:
        for metric in metrics:
            eval_metric = getattr(eval_model, metric)
            result = eval_metric(data_df, pred_df, k=k)
            result_dic[metric + "@" + str(k)] = result
    return result_dic


@timeit
def train_eval_worker(
    testEngine, valid_df, test_df, valid_pred, test_pred, epoch, TOP_K=10
):
    valid_result = evaluate(valid_df, valid_pred, testEngine.metrics, TOP_K)
    test_result = evaluate(test_df, test_pred, testEngine.metrics, TOP_K)
    lock_train_eval.acquire()  # need to be test
    testEngine.record_performance(valid_result, test_result, epoch)
    testEngine.expose_performance(valid_result, test_result)
    if (
        valid_result[testEngine.config["validate_metric"]]
        > testEngine.best_valid_performance
    ):
        testEngine.n_no_update = 0
        testEngine.best_performance = valid_result[testEngine.config["validate_metric"]]
        print(valid_result)
    else:
        testEngine.n_no_update += 1
    lock_train_eval.release()
    # lock record and get best performance
    return valid_result, test_result


@timeit
def test_eval_worker(testEngine, eval_data_df, prediction, TOP_K=[5, 10, 20]):
    """
    Prediction and evalution on test set
    """
    result_para = {
        "model": [testEngine.config["model"]],
        "dataset": [testEngine.config["dataset"]],
        "data_split": [testEngine.config["data_split"]],
        "temp_train": [testEngine.config["temp_train"]],
        "emb_dim": [int(testEngine.config["emb_dim"])],
        "lr": [testEngine.config["lr"]],
        "batch_size": [int(testEngine.config["batch_size"])],
        "optimizer": [testEngine.config["optimizer"]],
        "num_epoch": [testEngine.config["num_epoch"]],
        "model_run_id": [testEngine.config["model_run_id"]],
        "run_time": [testEngine.config["run_time"]],
    }
    if "late_dim" in testEngine.config:
        result_para["late_dim"] = [int(testEngine.config["late_dim"])]
    if "alpha" in testEngine.config:
        result_para["alpha"] = [int(testEngine.config["alpha"])]
    if "activator" in testEngine.config:
        result_para["activator"] = [testEngine.config["activator"]]

    test_result_dic = evaluate(eval_data_df, prediction, testEngine.metrics, TOP_K)
    test_result_dic.update(result_para)
    lock_test_eval.acquire()  # need to be test
    result_df = pd.DataFrame(test_result_dic)
    save_result(result_df, testEngine.config["result_file"])
    lock_test_eval.release()
    return test_result_dic


class EvalEngine(object):
    def __init__(self, config):
        self.config = config  # model configuration, should be a dic
        self.metrics = config["metrics"]
        self.validate_metric = config["validate_metric"]
        self.writer = SummaryWriter(log_dir=config["run_dir"])  # tensorboard writer
        self.writer.add_text(
            "model/config", dict2str(config), 0,
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

    def train_eval(self, valid_data_df, test_data_df, model, epoch_id=0, TOP_K=10):
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
                TOP_K,
            ),
        )
        worker.start()

    @timeit
    def test_eval(self, test_df_list, model, TOP_K=[5, 10, 20]):
        for i, test_data_df in enumerate(test_df_list):
            test_pred = self.predict(test_data_df, model)
            worker = Thread(
                target=test_eval_worker,
                args=(self, test_data_df, test_pred, TOP_K),
                name="test_{}".format(i),
            )
            worker.start()

    def record_performance(self, validata_result, test_result, epoch_id):
        for metric in self.metrics:
            self.writer.add_scalars(
                "performance/" + metric,
                {
                    "valid": validata_result[metric + "@10"],
                    "test": test_result[metric + "@10"],
                },
                epoch_id,
            )

    def init_prometheus_client(self):
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
        for metric in self.config["metrics"]:
            self.gauges_valid[metric].labels(*self.labels).set(
                valid_result[metric + "@" + str(10)]
            )
            self.gauges_test[metric].labels(*self.labels).set(
                test_result[metric + "@" + str(10)]
            )
