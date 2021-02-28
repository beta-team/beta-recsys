# coding=utf-8

"""This is the core implementation of the evaluation."""
import concurrent.futures
from threading import Lock, Thread

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm

from ..utils import evaluation as eval_model
from ..utils.common_util import print_dict_as_table, save_to_csv, timeit
from ..utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
)
from ..utils.seq_evaluation import mrr, ndcg, precision, recall

lock_train_eval = Lock()
lock_test_eval = Lock()


def computeRePos(time_seq, time_span):
    """Compute position matrix for a user.

    Args:
        time_seq ([type]): [description]
        time_span ([type]): [description]

    Returns:
        [type]: [description]
    """
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def evaluate(data_df, predictions, metrics, k_li):
    """Evaluate the performance of a prediction by different metrics.

    Args:
        data_df (DataFrame): the dataset to be evaluated.
        predictions (narray): 1-D array. The predicted scores for each user-item pair in the dataset.
        metrics (list):  metrics to be evaluated.
        k_li (int or list): top k (s) to be evaluated.

    Returns:
        result_dic (dict): Performance result.
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
    metric_mapping = {
        "rmse": eval_model.rmse,
        "mae": eval_model.mae,
        "rsquared": eval_model.rsquared,
        "ndcg": eval_model.ndcg_at_k,
        "map": eval_model.map_at_k,
        "precision": eval_model.precision_at_k,
        "recall": eval_model.recall_at_k,
    }

    result_dic = {}
    if type(k_li) != list:
        k_li = [k_li]
    for k in k_li:
        for metric in metrics:
            result = metric_mapping[metric](data_df, pred_df, k=k)
            result_dic[f"{metric}@{k}"] = result
    return result_dic


@timeit
def train_eval_worker(testEngine, valid_df, test_df, valid_pred, test_pred, epoch):
    """Start a worker for the evaluation during training.

    Args:
        testEngine:
        valid_df:
        test_df:
        valid_pred:
        test_pred:
        epoch (int):

    Returns:
        (dict,dict): dictionary with performances on validation and testing sets.
    """
    testEngine.n_worker += 1
    valid_result = evaluate(
        valid_df, valid_pred, testEngine.metrics, testEngine.valid_k
    )
    test_result = evaluate(test_df, test_pred, testEngine.metrics, testEngine.valid_k)
    lock_train_eval.acquire()
    testEngine.record_performance(valid_result, test_result, epoch)
    testEngine.expose_performance(valid_result, test_result)
    if (
        valid_result[f"{testEngine.valid_metric}@{testEngine.valid_k}"]
        > testEngine.best_valid_performance
    ):
        testEngine.n_no_update = 0
        print(
            "Current testEngine.best_valid_performance"
            f" {testEngine.best_valid_performance}"
        )
        testEngine.best_valid_performance = valid_result[
            f"{testEngine.valid_metric}@{testEngine.valid_k}"
        ]
        print_dict_as_table(
            valid_result,
            tag=f"performance on validation at epoch {epoch}",
            columns=["metrics", "values"],
        )
        print_dict_as_table(
            test_result,
            tag=f"performance on testing at epoch {epoch}",
            columns=["metrics", "values"],
        )
    else:
        testEngine.n_no_update += 1
        print(f"number of epochs that have no update {testEngine.n_no_update}")

    testEngine.n_worker -= 1
    lock_train_eval.release()
    # lock record and get best performance
    return valid_result, test_result


@timeit
def test_eval_worker(testEngine, eval_data_df, prediction):
    """Start a worker for the evaluation during training.

    Prediction and evaluation on the testing set.
    """
    config = testEngine.config["system"]
    result_para = {
        "run_time": [testEngine.config["run_time"]],
    }
    testEngine.n_worker += 1
    for cfg in ["model", "dataset"]:
        for col in testEngine.config[cfg]["result_col"]:
            result_para[col] = [testEngine.config[cfg][col]]

    test_result_dic = evaluate(
        eval_data_df, prediction, testEngine.metrics, testEngine.k
    )
    print_dict_as_table(
        test_result_dic,
        tag="performance on test",
        columns=["metrics", "values"],
    )
    test_result_dic.update(result_para)
    lock_test_eval.acquire()  # need to be test
    result_df = pd.DataFrame(test_result_dic)
    save_to_csv(result_df, config["result_file"])
    lock_test_eval.release()
    testEngine.n_worker -= 1
    save_mode = config["save_mode"]
    if save_mode == "per_user":
        rating_pre_df = eval_data_df.drop(columns=["col_timestamp"])
        rating_pre_df["col_prediction"] = prediction
        save_to_csv(
            rating_pre_df,
            config["root_dir"] + "/" + config["result_dir"] + config["model_run_id"],
        )
    else:
        pass
    return test_result_dic


class EvalEngine(object):
    """The base evaluation engine."""

    def __init__(self, config):
        """Init EvalEngine Class.

        Args:
            config (dict): parameters for the model
        """
        self.config = config  # model configuration, should be a dic
        self.metrics = config["system"]["metrics"]
        self.k = config["system"]["k"]
        self.valid_metric = config["system"]["valid_metric"]
        self.valid_k = config["system"]["valid_k"]
        self.batch_eval = (
            config["model"]["batch_eval"] if "batch_eval" in config else False
        )
        self.batch_size = config["model"]["batch_size"]
        self.writer = SummaryWriter(
            log_dir=config["model"]["run_dir"]
        )  # tensorboard writer
        self.writer.add_text(
            "config/system",
            pd.DataFrame(
                config["system"].items(), columns=["parameters", "values"]
            ).to_string(),
            0,
        )
        self.writer.add_text(
            "config/model",
            pd.DataFrame(
                config["model"].items(), columns=["parameters", "values"]
            ).to_string(),
            0,
        )
        self.n_worker = 0
        self.n_no_update = 0
        self.best_valid_performance = 0
        print("Initializing test engine ...")

    def flush(self):
        """Flush eval_engine."""
        self.n_no_update = 0
        self.best_valid_performance = 0

    def predict(self, data_df, model, batch_eval=False):
        """Make prediction for a trained model.

        Args:
            data_df (DataFrame): A dataset to be evaluated.
            model: A trained model.
            batch_eval (Boolean): A signal to indicate if the model is evaluated in batches.

        Returns:
            array: predicted scores.
        """
        user_ids = data_df[DEFAULT_USER_COL].to_numpy()
        item_ids = data_df[DEFAULT_ITEM_COL].to_numpy()
        if batch_eval:
            n_batch = len(data_df) // self.batch_size + 1
            predictions = np.array([])
            for idx in range(n_batch):
                start_idx = idx * self.batch_size
                end_idx = min((idx + 1) * self.batch_size, len(data_df))
                sub_user_ids = user_ids[start_idx:end_idx]
                sub_item_ids = item_ids[start_idx:end_idx]
                sub_predictions = np.array(
                    model.predict(sub_user_ids, sub_item_ids)
                    .flatten()
                    .to(torch.device("cpu"))
                    .detach()
                    .numpy()
                )
                predictions = np.append(predictions, sub_predictions)
        else:
            predictions = np.array(
                model.predict(user_ids, item_ids)
                .flatten()
                .to(torch.device("cpu"))
                .detach()
                .numpy()
            )
        return predictions

    def seq_predict(self, train_seq, data_df, model, maxlen):
        """Make prediction for a trained model.

        Args:
            data_df (DataFrame): A dataset to be evaluated.
            model: A trained model.
            batch_eval (Boolean): A signal to indicate if the model is evaluated in batches.

        Returns:
            array: predicted scores.
        """
        user_ids = data_df[DEFAULT_USER_COL].to_numpy()
        item_ids = data_df[DEFAULT_ITEM_COL].to_numpy()
        user_item_list = data_df.groupby([DEFAULT_USER_COL])[DEFAULT_ITEM_COL].apply(
            list
        )
        result_dic = {}
        with torch.no_grad():
            for u, items in user_item_list.items():
                if len(train_seq[u]) < 1:
                    continue
                seq = np.zeros([maxlen], dtype=np.int32)
                idx = maxlen - 1
                for i in reversed(train_seq[u]):
                    seq[idx] = i
                    idx -= 1
                    if idx == -1:
                        break
                score = model.predict(
                    np.array([u]),
                    np.array([seq]),
                    np.array(items),
                )
                score = np.array(
                    score.flatten().to(torch.device("cpu")).detach().numpy() * -1
                )
                for i, item in enumerate(items):
                    result_dic[(u, item)] = score[i]
        predictions = []
        for u, i in zip(user_ids, item_ids):
            predictions.append(result_dic[(u, i)])
        return np.array(predictions)

    def test_seq_predict(self, train_seq, valid_data_df, test_data_df, model, maxlen):
        """Make prediction for a trained model.

        Args:
            data_df (DataFrame): A dataset to be evaluated.
            model: A trained model.
            batch_eval (Boolean): A signal to indicate if the model is evaluated in batches.

        Returns:
            array: predicted scores.
        """
        valid_user_item_list = (
            valid_data_df[valid_data_df[DEFAULT_RATING_COL] > 0]
            .groupby([DEFAULT_USER_COL])[DEFAULT_ITEM_COL]
            .apply(list)
        )
        user_item_list = test_data_df.groupby([DEFAULT_USER_COL])[
            DEFAULT_ITEM_COL
        ].apply(list)
        user_ids = test_data_df[DEFAULT_USER_COL].to_numpy()
        item_ids = test_data_df[DEFAULT_ITEM_COL].to_numpy()
        predictions = []
        result_dic = {}
        with torch.no_grad():
            for u, items in user_item_list.items():
                if len(train_seq[u]) < 1:
                    continue
                seq = np.zeros([maxlen], dtype=np.int32)
                idx = maxlen - 1
                if u in valid_user_item_list:
                    for i in reversed(
                        valid_user_item_list[u]
                    ):  # add validate item list
                        seq[idx] = i
                        idx -= 1
                        if idx == -1:
                            break
                for i in reversed(train_seq[u]):
                    if idx <= -1:
                        break
                    seq[idx] = i
                    idx -= 1
                    if idx == -1:
                        break
                score = model.predict(
                    np.array([u]),
                    np.array([seq]),
                    np.array(items),
                )
                score = np.array(
                    score.flatten().to(torch.device("cpu")).detach().numpy() * -1
                )
                for i, item in enumerate(items):
                    result_dic[(u, item)] = score[i]
        for u, i in zip(user_ids, item_ids):
            predictions.append(result_dic[(u, i)])
        return np.array(predictions)

    # implemented this due to differences in indexing for TiSASRec, not sure if needed at all
    def seq_predict_time(self, train_seq, data_df, model, maxlen, time_span):
        """Make prediction for a trained model.

        Args:
            data_df (DataFrame): A dataset to be evaluated.
            model: A trained model.
            batch_eval (Boolean): A signal to indicate if the model is evaluated in batches.

        Returns:
            array: predicted scores.
        """
        user_ids = data_df[DEFAULT_USER_COL].to_numpy()
        item_ids = data_df[DEFAULT_ITEM_COL].to_numpy()
        user_item_list = data_df.groupby([DEFAULT_USER_COL])[DEFAULT_ITEM_COL].apply(
            list
        )
        result_dic = {}
        with torch.no_grad():
            print("Train seq", type(train_seq))
            for u, items in user_item_list.items():
                if len(train_seq[u]) < 1:
                    continue
                seq = np.zeros([maxlen], dtype=np.int32)
                time_seq = np.zeros([maxlen], dtype=np.int32)
                time_matrix = computeRePos(time_seq, time_span)
                idx = maxlen - 1
                for i in reversed(train_seq[u]):
                    seq[idx] = i[0]
                    time_seq[idx] = i[1]
                    idx -= 1
                    if idx == -1:
                        break
                score = model.predict(
                    np.array([u]),
                    np.array([seq]),
                    np.array([time_matrix]),
                    np.array(items),
                )
                score = np.array(
                    score.flatten().to(torch.device("cpu")).detach().numpy() * -1
                )
                for i, item in enumerate(items):
                    result_dic[(u, item)] = score[i]
        predictions = []
        for u, i in zip(user_ids, item_ids):
            predictions.append(result_dic[(u, i)])
        return np.array(predictions)

    # implemented this due to differences in indexing, not sure if needed
    def test_seq_predict_time(
        self, train_seq, valid_data_df, test_data_df, model, maxlen, time_span
    ):
        """Make prediction for a trained model.

        Args:
            data_df (DataFrame): A dataset to be evaluated.
            model: A trained model.
            batch_eval (Boolean): A signal to indicate if the model is evaluated in batches.

        Returns:
            array: predicted scores.
        """
        valid_user_item_list = (
            valid_data_df[valid_data_df[DEFAULT_RATING_COL] > 0]
            .groupby([DEFAULT_USER_COL])[DEFAULT_ITEM_COL]
            .apply(list)
        )
        user_item_list = test_data_df.groupby([DEFAULT_USER_COL])[
            DEFAULT_ITEM_COL
        ].apply(list)
        user_ids = test_data_df[DEFAULT_USER_COL].to_numpy()
        item_ids = test_data_df[DEFAULT_ITEM_COL].to_numpy()
        predictions = []
        result_dic = {}
        with torch.no_grad():
            for u, items in user_item_list.items():
                if len(train_seq[u]) < 1:
                    continue
                seq = np.zeros([maxlen], dtype=np.int32)
                time_seq = np.zeros([maxlen], dtype=np.int32)
                time_matrix = computeRePos(time_seq, time_span)
                idx = maxlen - 1
                if u in valid_user_item_list:
                    for i in reversed(
                        valid_user_item_list[u]
                    ):  # add validate item list
                        seq[idx] = i[0]
                        time_seq[idx] = i[1]
                        idx -= 1
                        if idx == -1:
                            break
                for i in reversed(train_seq[u]):
                    if idx <= -1:
                        break
                    seq[idx] = i[0]
                    time_seq[idx] = i[1]
                    idx -= 1
                    if idx == -1:
                        break
                score = model.predict(
                    np.array([u]),
                    np.array([seq]),
                    np.array([time_matrix]),
                    np.array(items),
                )
                score = np.array(
                    score.flatten().to(torch.device("cpu")).detach().numpy() * -1
                )
                for i, item in enumerate(items):
                    result_dic[(u, item)] = score[i]
        for u, i in zip(user_ids, item_ids):
            predictions.append(result_dic[(u, i)])
        return np.array(predictions)

    def train_eval(self, valid_data_df, test_data_df, model, epoch_id=0):
        """Evaluate the performance for a (validation) dataset with multiThread.

        Args:
            valid_data_df (DataFrame): A validation dataset.
            test_data_df (DataFrame): A testing dataset.
            model: trained model.
            epoch_id: epoch_id.
            k (int or list): top k result to be evaluate.
        """
        valid_pred = self.predict(valid_data_df, model, self.batch_eval)
        test_pred = self.predict(test_data_df, model, self.batch_eval)
        worker = Thread(
            target=train_eval_worker,
            args=(
                self,
                valid_data_df,
                test_data_df,
                valid_pred,
                test_pred,
                epoch_id,
            ),
        )
        worker.start()

    def seq_train_eval(
        self, train_seq, valid_data_df, test_data_df, model, maxlen, epoch_id=0
    ):
        """Evaluate the performance for a (validation) dataset with multiThread.

        Args:
            valid_data_df (DataFrame): A validation dataset.
            test_data_df (DataFrame): A testing dataset.
            model: trained model.
            epoch_id: epoch_id.
            k (int or list): top k result to be evaluate.
        """
        valid_pred = self.seq_predict(train_seq, valid_data_df, model, maxlen)
        test_pred = self.test_seq_predict(
            train_seq, valid_data_df, test_data_df, model, maxlen
        )
        worker = Thread(
            target=train_eval_worker,
            args=(
                self,
                valid_data_df,
                test_data_df,
                valid_pred,
                test_pred,
                epoch_id,
            ),
        )
        worker.start()

    # implemented this due to differences in indexing, not sure if needed
    def seq_train_eval_time(
        self,
        train_seq,
        valid_data_df,
        test_data_df,
        model,
        maxlen,
        time_span,
        epoch_id=0,
    ):
        """Evaluate the performance for a (validation) dataset with multiThread.

        Args:
            valid_data_df (DataFrame): A validation dataset.
            test_data_df (DataFrame): A testing dataset.
            model: trained model.
            epoch_id: epoch_id.
            k (int or list): top k result to be evaluate.
        """
        valid_pred = self.seq_predict_time(
            train_seq, valid_data_df, model, maxlen, time_span
        )
        test_pred = self.test_seq_predict_time(
            train_seq, valid_data_df, test_data_df, model, maxlen, time_span
        )
        worker = Thread(
            target=train_eval_worker,
            args=(
                self,
                valid_data_df,
                test_data_df,
                valid_pred,
                test_pred,
                epoch_id,
            ),
        )
        worker.start()

    @timeit
    def test_eval(self, test_df_list, model):
        """Evaluate the performance for a (testing) dataset list with multiThread.

        Args:
            test_df_list (list): (testing) dataset list.
            model: trained model.
        """
        if type(test_df_list) is not list:  # compatible for testing a single test set
            test_df_list = [test_df_list]
        return_value_list = []
        for i, test_data_df in enumerate(test_df_list):
            test_pred = self.predict(test_data_df, model, self.batch_eval)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    test_eval_worker, self, test_data_df, test_pred
                )
                return_value = future.result()
                return_value_list.append(return_value)
        return return_value_list

    def record_performance(self, valid_result, test_result, epoch_id):
        """Record perforance result on tensorboard.

        Args:
            valid_result (dict): Performance result of validation set.
            test_result (dict): Performance result of testing set.
            epoch_id (int): epoch_id.
        """
        for metric in self.metrics:
            self.writer.add_scalars(
                "performance/" + metric,
                {
                    "valid": valid_result[f"{metric}@{self.valid_k}"],
                    "test": test_result[f"{metric}@{self.valid_k}"],
                },
                epoch_id,
            )


class SeqEvalEngine(object):
    """The base evaluation engine for sequential recommendation."""

    def __init__(self, config):
        """Init SeqEvalEngine Class.

        Args:
            config (dict): parameters for the model.
        """
        self.config = config  # model configuration, should be a dic
        self.metrics = config["system"]["metrics"]
        self.valid_metric = config["system"]["valid_metric"]

    def sequential_evaluation(
        self,
        recommender,
        test_sequences,
        evaluation_functions,
        users=None,
        given_k=1,
        look_ahead=1,
        top_n=10,
        scroll=True,
        step=1,
    ):
        """Run sequential evaluation of a recommender over a set of test sequences.

        Args:
            recommender (object): the instance of the recommender to test.
            test_sequences (List): the set of test sequences
            evaluation_functions (dict): list of evaluation metric functions.
            users (List): (optional) the list of user ids associated to each test sequence.
            given_k (int): (optional) the initial size of each user profile, starting from
                        the first interaction in the sequence.
                        If <0, start counting from the end of the sequence. It must be != 0.
            look_ahead (int): (optional) number of subsequent interactions in the sequence to be considered as ground truth.
                        It can be any positive number or 'all' to extend the ground truth until the end of the sequence.
            top_n (int): (optional) size of the recommendation list
            scroll (boolean): (optional) whether to scroll the ground truth until the end of the sequence.
                    If True, expand the user profile and move the ground truth forward of `step` interactions.
                    Recompute and evaluate recommendations every time.
                    If False, evaluate recommendations once per sequence without expanding the user profile.
            step (int): (optional) number of interactions that will be added to the user profile at each
                        step of the sequential evaluation.

        Returns:
            metrics/len(test_sequences) (1d array): the list of the average values for each evaluation metric.
        """
        if given_k == 0:
            raise ValueError("given_k must be != 0")

        metrics = np.zeros(len(evaluation_functions))
        with tqdm(total=len(test_sequences)) as pbar:
            for i, test_seq in enumerate(test_sequences):
                if users is not None:
                    user = users[i]
                else:
                    user = None
                if scroll:
                    metrics += self.sequence_sequential_evaluation(
                        recommender,
                        test_seq,
                        evaluation_functions,
                        user,
                        given_k,
                        look_ahead,
                        top_n,
                        step,
                    )
                else:
                    metrics += self.evaluate_sequence(
                        recommender,
                        test_seq,
                        evaluation_functions,
                        user,
                        given_k,
                        look_ahead,
                        top_n,
                    )
                pbar.update(1)
        return metrics / len(test_sequences)

    def evaluate_sequence(
        self,
        recommender,
        seq,
        evaluation_functions,
        user,
        given_k,
        look_ahead,
        top_n,
    ):
        """Compute metrics for each sequence.

        Args:
            recommender (object): which recommender to use
            seq (List): the user_profile/ context
            given_k (int): last element used as ground truth. NB if <0 it is interpreted as first elements to keep
            evaluation_functions (dict): which function to use to evaluate the rec performance
            look_ahead (int): number of elements in ground truth to consider.
                        If look_ahead = 'all' then all the ground_truth sequence is considered

        Returns:
            np.array(tmp_results) (1d array): performance of recommender.
        """
        # safety checks
        if given_k < 0:
            given_k = len(seq) + given_k

        user_profile = seq[:given_k]
        ground_truth = seq[given_k:]

        # restrict ground truth to look_ahead
        ground_truth = (
            ground_truth[:look_ahead] if look_ahead != "all" else ground_truth
        )
        ground_truth = list(map(lambda x: [x], ground_truth))  # list of list format

        if len(user_profile) == 0 or len(ground_truth) == 0:
            # if any of the two missing all evaluation functions are 0
            return np.zeros(len(evaluation_functions))

        r = recommender.recommend(user_profile, user)[:top_n]

        if not r:
            # no recommendation found
            return np.zeros(len(evaluation_functions))
        reco_list = recommender.get_recommendation_list(r)

        tmp_results = []
        for f in evaluation_functions:
            tmp_results.append(f(ground_truth, reco_list))
        return np.array(tmp_results)

    def sequence_sequential_evaluation(
        self,
        recommender,
        seq,
        evaluation_functions,
        user,
        given_k,
        look_ahead,
        top_n,
        step,
    ):
        """Compute metrics for each sequence incrementally.

        Args:
            recommender (object): which recommender to use
            seq (List): the user_profile/ context
            given_k (int): last element used as ground truth. NB if <0 it is interpreted as first elements to keep
            evaluation_functions (dict): which function to use to evaluate the rec performance
            look_ahead (int): number of elements in ground truth to consider.
                            If look_ahead = 'all' then all the ground_truth sequence is considered

        Returns:
            eval_res/eval_cnt (1d array): performance of recommender.
        """
        if given_k < 0:
            given_k = len(seq) + given_k

        eval_res = 0.0
        eval_cnt = 0
        for gk in range(given_k, len(seq), step):
            eval_res += self.evaluate_sequence(
                recommender,
                seq,
                evaluation_functions,
                user,
                gk,
                look_ahead,
                top_n,
            )
            eval_cnt += 1
        return eval_res / eval_cnt

    def get_test_sequences(self, test_data, given_k):
        """Run evaluation only over sequences longer than abs(LAST_K).

        Args:
            test_data (pandas.DataFrame): Test set.
            given_k (int): last element used as ground truth.

        Returns:
            test_sequences (List): list of sequences for testing.

        """
        # we can run evaluation only over sequences longer than abs(LAST_K)
        test_sequences = test_data.loc[
            test_data["col_sequence"].map(len) > abs(given_k), "col_sequence"
        ].values
        return test_sequences

    def train_eval_seq(self, valid_data, test_data, recommender, epoch_id=0):
        """Compute performance of the sequential models with validation and test datasets for each epoch during training.

        Args:
            valid_data (pandas.DataFrame): validation dataset.
            test_data (pandas.DataFrame): test dataset.
            recommender (Object): Sequential recommender.
            epoch_id (int): id of the epoch.
            k (int): size of the recommendation list

        Returns:
            None

        """
        METRICS = {
            "ndcg": ndcg,
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
        }
        TOPN = self.config["system"]["valid_k"]  # length of the recommendation list

        # GIVEN_K=-1, LOOK_AHEAD=1, STEP=1 corresponds to the classical next-item evaluation
        GIVEN_K = self.config["model"]["GIVEN_K"]
        LOOK_AHEAD = self.config["model"]["LOOK_AHEAD"]
        STEP = self.config["model"]["STEP"]
        scroll = self.config["model"]["scroll"]

        # valid data
        valid_sequences = self.get_test_sequences(valid_data, GIVEN_K)
        print("{} sequences available for evaluation".format(len(valid_sequences)))

        valid_results = self.sequential_evaluation(
            recommender,
            test_sequences=valid_sequences,
            given_k=GIVEN_K,
            look_ahead=LOOK_AHEAD,
            evaluation_functions=METRICS.values(),
            top_n=TOPN,
            scroll=scroll,  # scrolling averages metrics over all profile lengths
            step=STEP,
        )

        print(
            "Sequential evaluation (GIVEN_K={}, LOOK_AHEAD={}, STEP={})".format(
                GIVEN_K, LOOK_AHEAD, STEP
            )
        )
        for mname, mvalue in zip(METRICS.keys(), valid_results):
            print("\t{}@{}: {:.4f}".format(mname, TOPN, mvalue))

        # test data
        test_sequences = self.get_test_sequences(test_data, GIVEN_K)
        print("{} sequences available for evaluation".format(len(test_sequences)))

        test_results = self.sequential_evaluation(
            recommender,
            test_sequences=test_sequences,
            given_k=GIVEN_K,
            look_ahead=LOOK_AHEAD,
            evaluation_functions=METRICS.values(),
            top_n=TOPN,
            scroll=scroll,  # scrolling averages metrics over all profile lengths
            step=STEP,
        )

        print(
            "Sequential evaluation (GIVEN_K={}, LOOK_AHEAD={}, STEP={})".format(
                GIVEN_K, LOOK_AHEAD, STEP
            )
        )
        for mname, mvalue in zip(METRICS.keys(), test_results):
            print("\t{}@{}: {:.4f}".format(mname, TOPN, mvalue))

    def test_eval_seq(self, test_data, recommender):
        """Compute performance of the sequential models with test dataset.

        Args:
            test_data (pandas.DataFrame): test dataset.
            recommender (Object): Sequential recommender.
            k (int): size of the recommendation list

        Returns:
            None
        """
        METRICS = {
            "ndcg": ndcg,
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
        }
        TOPNs = self.config["system"]["k"]  # length of the recommendation list

        # GIVEN_K=-1, LOOK_AHEAD=1, STEP=1 corresponds to the classical next-item evaluation
        GIVEN_K = self.config["model"]["GIVEN_K"]
        LOOK_AHEAD = self.config["model"]["LOOK_AHEAD"]
        STEP = self.config["model"]["STEP"]
        scroll = self.config["model"]["scroll"]

        # test data
        test_sequences = self.get_test_sequences(test_data, GIVEN_K)
        print("{} sequences available for evaluation".format(len(test_sequences)))

        for TOPN in TOPNs:
            test_results = self.sequential_evaluation(
                recommender,
                test_sequences=test_sequences,
                given_k=GIVEN_K,
                look_ahead=LOOK_AHEAD,
                evaluation_functions=METRICS.values(),
                top_n=TOPN,
                scroll=scroll,  # scrolling averages metrics over all profile lengths
                step=STEP,
            )

            print(
                "Sequential evaluation (GIVEN_K={}, LOOK_AHEAD={}, STEP={})".format(
                    GIVEN_K, LOOK_AHEAD, STEP
                )
            )
            for mname, mvalue in zip(METRICS.keys(), test_results):
                print("\t{}@{}: {:.4f}".format(mname, TOPN, mvalue))
