import os
import pickle
import time
from collections import defaultdict
from multiprocessing import Process, Queue

import numpy as np
from munch import munchify
from ray import tune
from tqdm import tqdm

from ..core.recommender import Recommender
from ..models.tisasrec import TiSASRecEngine
from ..utils.monitor import Monitor


# same as SASRec
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


# new in TiSASRec
def timeSlice(time_set):
    """Normalize timestamps.

    Args:
        time_set ([type]): [description]

    Returns:
        [type]: [description]
    """
    time_min = min(time_set)
    time_map = dict()
    for t in time_set:  # float as map key?
        time_map[t] = int(round(float(t - time_min)))
    return time_map


# new in TiSASRec
def cleanAndsort(User, time_map):
    """Get user, lengths of users and items and timestamps.

    Args:
        User ([type]): [description]
        time_map ([type]): [description]

    Returns:
        [type]: [description]
    """
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(
            map(lambda x: [item_map[x[0]], time_map[x[1]]], items)
        )

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(
            map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1)], items)
        )
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)


# new in TiSASRec
def computeRePos(time_seq, time_span):
    """Compute position matrix for a single user.

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


# new in TiSASRec
def Relation(user_train, usernum, maxlen, time_span):
    """Compute relation matrix for all users.

    Args:
        user_train ([type]): [description]
        usernum ([type]): [description]
        maxlen ([type]): [description]
        time_span ([type]): [description]

    Returns:
        [type]: [description]
    """
    data_train = dict()
    for user in tqdm(range(1, usernum + 1), desc="Preparing relation matrix"):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1:
                break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train


# there's a similar function in SASRec but I'm not sure what code in Beta-Recsys
# corresponds to this
def data_partition():
    """Prepare and split data.

    Returns:
        [type]: [description]
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    print("Preparing data...")
    f = open("/beta-recsys/ml-1m.txt")
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split("\t")
        except ValueError:
            u, i, timestamp = line.rstrip().split("\t")
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open("/beta-recsys/ml-1m.txt")
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split("\t")
        except ValueError:  # Unconfirmed Error
            u, i, timestamp = line.rstrip().split("\t")
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        if user_count[u] < 5 or item_count[i] < 5:  # hard-coded
            continue
        time_set.add(timestamp)
        User[u].append([i, timestamp])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    print("Preparing done...")
    return [user_train, user_valid, user_test, usernum, itemnum]


def sample_function(
    user_train,
    usernum,
    itemnum,
    batch_size,
    maxlen,
    relation_matrix,
    result_queue,
    SEED,
):
    """Sample batch of pos and neg sequences.

    Args:
        user_train ([type]): [description]
        usernum ([type]): [description]
        itemnum ([type]): [description]
        batch_size ([type]): [description]
        maxlen ([type]): [description]
        relation_matrix ([type]): [description]
        result_queue ([type]): [description]
        SEED ([type]): [description]
    """

    def sample(user):
        """Sample for a single user.

        Args:
            user ([type]): [description]
        Returns:
            [type]: [description]

        """
        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1][0]

        idx = maxlen - 1
        ts = set(map(lambda x: x[0], user_train[user]))
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1:
                break
        time_matrix = relation_matrix[user]
        return (user, seq, time_seq, time_matrix, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1:
                user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    """MultiThread Sampler.

    Args:
        object ([type]): [description]
    """

    def __init__(
        self,
        User,
        usernum,
        itemnum,
        relation_matrix,
        batch_size=64,
        maxlen=10,
        n_workers=1,
    ):
        """Initialize workers.

        Args:
            User ([type]): [description]
            usernum ([type]): [description]
            itemnum ([type]): [description]
            relation_matrix ([type]): [description]
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
                        relation_matrix,
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
    train_engine = TiSASRec(munchify(config))
    result = train_engine.train(data)
    while train_engine.eval_engine.n_worker > 0:
        time.sleep(20)
    tune.report(
        valid_metric=result["valid_metric"],
        model_save_dir=result["model_save_dir"],
    )


class TiSASRec(Recommender):
    """The TiSASRec Model."""

    def __init__(self, config):
        """Initialize the config of this recommender.

        Args:
            config:
        """
        super(TiSASRec, self).__init__(config, name="TiSASRec")

    def init_engine(self, data):
        """Initialize the required parameters for the model.

        Args:
            data: the Dataset object.

        """
        self.config["model"]["n_users"] = data.n_users
        self.config["model"]["n_items"] = data.n_items
        self.engine = TiSASRecEngine(self.config)

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
        self.engine = TiSASRecEngine(self.config)
        self.engine.data = data
        data.config = self.config
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        [user_train, user_valid, user_test, usernum, itemnum] = data_partition()
        # from main.py in original TiSASRec implementation
        try:
            self.relation_matrix = pickle.load(
                open(
                    "/beta-recsys/results/relation_matrix_%s_%d_%d.pickle"
                    % (
                        self.config["dataset"]["dataset"],
                        self.config["model"]["maxlen"],
                        self.config["model"]["time_span"],
                    ),
                    "rb",
                )
            )
        except FileNotFoundError:
            self.relation_matrix = Relation(
                user_train,
                usernum,
                self.config["model"]["maxlen"],
                self.config["model"]["time_span"],
            )
            pickle.dump(
                self.relation_matrix,
                open(
                    "/beta-recsys/results/relation_matrix_%s_%d_%d.pickle"
                    % (
                        self.config["dataset"]["dataset"],
                        self.config["model"]["maxlen"],
                        self.config["model"]["time_span"],
                    ),
                    "wb",
                ),
            )
        # tried data.get_train_seq() as well but not sure if it accounts
        # for the different format of data compared with SASRec
        sampler = WarpSampler(
            user_train,
            data.n_users,
            data.n_items,
            relation_matrix=self.relation_matrix,
            batch_size=self.config["model"]["batch_size"],
            maxlen=self.config["model"]["maxlen"],
            n_workers=3,
        )

        self.model_save_dir = os.path.join(
            self.config["system"]["model_save_dir"], self.config["model"]["save_name"]
        )
        # implemented a new function here due to differences in indexing
        # not sure if this is not needed and the original seq_train function could be used instead
        self._seq_train_time(
            engine=self.engine,
            train_loader=sampler,
            save_dir=self.model_save_dir,
            train_seq=user_train,  # not sure about what to use here
            valid_df=data.valid[0],
            test_df=data.test[0],
        )
        self.config["run_time"] = self.monitor.stop()
        return {
            "valid_metric": self.eval_engine.best_valid_performance,
            "model_save_dir": self.model_save_dir,
        }
