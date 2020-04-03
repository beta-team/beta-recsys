"""
Created on Feb 5, 2020 by Zaiqiao

Training engine for pytorch models.
Code was modified from the orginal codes of https://github.com/yihong-chen/neural-collaborative-filtering

@author: Zaiqiao Meng (zaiqiao.meng@gmail.com)

"""
import sys

sys.path.append("../")
import numpy as np
import pandas as pd
import GPUtil
import torch
from tensorboardX import SummaryWriter
import src.utils.evaluation as eval_model
import src.utils.constants as Constants


def dict2str(dic):
    dic_str = (
        "Configs: \n"
        + "\n".join([str(k) + ":\t" + str(v) for k, v in dic.items()])
        + "\n"
    )
    print(dic_str)
    return dic_str


class Engine(object):
    """Meta Engine for training & evaluating NCF model
    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration, should be a dic
        self.metrics = config["metrics"]
        # evaluation metrice list, options are
        # ['rmse', 'mae', 'rsquared', 'exp_var', 'auc', 'map_at_k', 'ndcg_at_k', 'precision_at_k', 'recall_at_k']
        self.set_device()
        self.set_optimizer()
        self.model.to(self.device)
        print(self.model)
        self.writer = SummaryWriter(log_dir=config["run_dir"])  # tensorboard writer
        self.writer.add_text(
            "model/config", dict2str(config), 0,
        )
        self.loss = torch.nn.BCELoss()

    def set_optimizer(self):
        if self.config["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.config["lr"],
            )
        elif self.config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config["lr"],
            )
        elif self.config["optimizer"] == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(), lr=self.config["lr"],
            )

    def set_device(self):
        self.device = torch.device(self.config["device_str"])
        self.model.device = self.device
        print("Setting device for torch_engine", self.device)

    def train_single_batch(self, batch_data, ratings=None):
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.optimizer.zero_grad()
        ratings_pred = self.model.forward(batch_data)
        loss = self.model.loss(ratings_pred.view(-1), ratings)
        loss.backward()
        self.model.optimizer.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        for batch_id, batch_data in enumerate(train_loader):
            assert isinstance(batch_data, torch.LongTensor)
            loss = self.train_single_batch(batch_data)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, total_loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def evaluate(self, eval_data_df, epoch_id=0, TOP_K=10):
        """ 
        evaluate the performance for all the eval_data_df.


        Parameters:
                eval_data_df: Dataframe with column naming DEFAULT_USER_COL, DEFAULT_ITEM_COL 
                and DEFAULT_RATING_COL
                TOP_K: if not specified, DEFAULT_K =10

        Returns:
                return the evaluation scores of the following metrics scores:MAP,NDCG,
                Precision,Recall on value of k.
                
                example:
                {'map_at_k': 0.5,
                 'NDCG@5': 0.5,
                 'Precision@5': 0.5,
                 'Recall@5':0.5
                 }
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        user_ids = eval_data_df[Constants.DEFAULT_USER_COL].to_numpy()
        item_ids = eval_data_df[Constants.DEFAULT_ITEM_COL].to_numpy()
        ratings = eval_data_df[Constants.DEFAULT_RATING_COL].to_numpy()
        prediction = np.array(
            self.model.predict(user_ids, item_ids)
            .flatten()
            .to(torch.device("cpu"))
            .detach()
            .numpy()
        )

        pred_df = pd.DataFrame(
            {
                Constants.DEFAULT_USER_COL: user_ids,
                Constants.DEFAULT_ITEM_COL: item_ids,
                Constants.DEFAULT_PREDICTION_COL: prediction,
            }
        )

        result_dic = {}
        if type(TOP_K) != list:
            TOP_K = [TOP_K]
        if 10 not in TOP_K:
            TOP_K.append(10)
        TOP_K = [5, 10, 20]
        for k in TOP_K:
            for metric in self.metrics:
                eval_metric = getattr(eval_model, metric)
                result = eval_metric(eval_data_df, pred_df, k=k)
                result_dic[metric + "@" + str(k)] = result
        return result_dic

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

    def save_checkpoint(self, model_dir):
        assert hasattr(self, "model"), "Please specify the exact model !"
        torch.save(self.model.state_dict(), model_dir)

    # to do
    def resume_checkpoint(self, model_dir, model=None):
        assert hasattr(self, "model"), "Please specify the exact model !"
        print("loading model from:", model_dir)
        state_dict = torch.load(
            model_dir, map_location=self.device
        )  # ensure all storage are on gpu
        if model == None:
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            return self.model
        else:
            model.load_state_dict(state_dict)
            model.to(self.device)
            return model