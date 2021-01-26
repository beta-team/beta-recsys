import os
import time
from multiprocessing import Process, Queue

import numpy as np
from munch import munchify
from ray import tune

from ..core.recommender import Recommender
from ..models.sasrec import SASRecEngine
from ..utils.monitor import Monitor


def random_neq(low, r, s):
    """Sampler for batch generation.

    Args:
        low ([type]): [description]
        r ([type]): [description]
        s ([type]): [description]

    Returns:
        [type]: [description]
    """
    t = np.random.randint(low, r)
    while t in s:
        t = np.random.randint(low, r)
    return t


def sample_function(
    user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED
):
    """Sample batch of pos and neg sequences.

    Args:
        user_train ([type]): [description]
        usernum ([type]): [description]
        itemnum ([type]): [description]
        batch_size ([type]): [description]
        maxlen ([type]): [description]
        result_queue ([type]): [description]
        SEED ([type]): [description]
    """

    def sample():

        user = np.random.randint(0, usernum)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(0, itemnum, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    """MultiThread Sampler.

    Args:
        object ([type]): [description]
    """

    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        """Initialize workers.

        Args:
            User ([type]): [description]
            usernum ([type]): [description]
            itemnum ([type]): [description]
            batch_size (int, optional): [description]. Defaults to 64.
            maxlen (int, optional): [description]. Defaults to 10.
            n_workers (int, optional): [description]. Defaults to 1.
        """
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(
                        User,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        self.result_queue,
                        np.random.randint(2e9),
                    ),
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        """Get next batch.

        Returns:
            [type]: [description]
        """
        return self.result_queue.get()

    def close(self):
        """Close processors."""
        for p in self.processors:
            p.terminate()
            p.join()


def tune_train(config):
    """Train the model with a hypyer-parameter tuner (ray).

    Args:
        config (dict): All the parameters for the model.
    """
    data = config["data"]
    train_engine = SASRec(munchify(config))
    result = train_engine.train(data)
    while train_engine.eval_engine.n_worker > 0:
        time.sleep(20)
    tune.report(
        valid_metric=result["valid_metric"],
        model_save_dir=result["model_save_dir"],
    )


class SASRec(Recommender):
    """The SASRec Model."""

    def __init__(self, config):
        """Initialize the config of this recommender.

        Args:
            config:
        """
        super(SASRec, self).__init__(config, name="SASRec")

    def init_engine(self, data):
        """Initialize the required parameters for the model.

        Args:
            data: the Dataset object.

        """
        self.config["model"]["n_users"] = data.n_users
        self.config["model"]["n_items"] = data.n_items
        self.engine = SASRecEngine(self.config)

    def train(self, data):
        """Training the model.

        Args:
            data: the Dataset object.

        Returns:
            dict: save k,v for "best_valid_performance" and "model_save_dir"

        """
        if ("tune" in self.args) and (self.args["tune"]):  # Tune the model.
            self.args.data = data
            tune_result = self.tune(tune_train)
            best_result = tune_result.loc[tune_result["valid_metric"].idxmax()]
            return {
                "valid_metric": best_result["valid_metric"],
                "model_save_dir": best_result["model_save_dir"],
            }

        self.gpu_id, self.config["device_str"] = self.get_device()  # Train the model.

        self.config["model"]["n_users"] = data.n_users
        self.config["model"]["n_items"] = data.n_items
        self.engine = SASRecEngine(self.config)
        self.engine.data = data
        data.config = self.config
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        sampler = WarpSampler(
            data.get_train_seq(),
            data.n_users,
            data.n_items,
            batch_size=self.config["model"]["batch_size"],
            maxlen=self.config["model"]["maxlen"],
            n_workers=3,
        )

        self.model_save_dir = os.path.join(
            self.config["system"]["model_save_dir"], self.config["model"]["save_name"]
        )
        self._seq_train(
            engine=self.engine,
            train_loader=sampler,
            save_dir=self.model_save_dir,
            train_seq=data.get_train_seq(),
            valid_df=data.valid[0],
            test_df=data.test[0],
        )
        self.config["run_time"] = self.monitor.stop()
        return {
            "valid_metric": self.eval_engine.best_valid_performance,
            "model_save_dir": self.model_save_dir,
        }
