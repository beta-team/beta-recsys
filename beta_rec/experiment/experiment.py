# coding=utf-8

"""
This is the implementation of experimental pipeline.

This class is still under development.
"""
import pandas as pd
from tabulate import tabulate


def print_result_as_table(results, tag=None):
    """Print results as a table."""
    eval_infos = set()
    for result in results:
        eval_infos.update(result.keys())
    eval_infos = list(eval_infos)
    print("-" * 80)
    if tag is not None:
        print(tag)
    for result in results:
        for eval_info in eval_infos:
            if eval_info not in result:
                result["eval_info"] = "--"
    df = pd.DataFrame(results)
    df = df.set_index("model")
    df = df.T
    print(tabulate(df, headers=df.columns, tablefmt="psql"))
    print("-" * 80)


class Experiment:
    """This enables the flow of an experiment with the beta-rec platform.

    Args:
    datasets: array of :obj: '<beta_rec.datasets>', required
            the experimental datasets (e.g. MovieLens)
    eval_methods: : array of string, required
            the evaluation method (e.g. ['load_leave_one_out'])
    models: array of :obj:`<beta_rec.recommenders>`, required
        A collection of recommender models to evaluate, e.g., [MF, GCN].
    metrics: array of string, default: None and every model has its default
        evaluation metrics in the configuration file.
        A collection of metrics to use to evaluate all the recommender
        models, e.g., ['ndcg', 'precision', 'recall'].
    eval_score: array of integer, default: None and every model has its default
        evaluation score in the configuration file.
        A list integer values to define evaluation scope on, \
        e.g., [1, 10, 20].
    model_dir: str, optional, default: None
        Path to a directory for loading a pretrained model
    save_dir: str, optional, default: None
        Path to a directory for storing trained models and logs. If None,
        models will NOT be stored and logs will be saved in the current
        working directory.
    result_file: str, optional, default: None and every model will be saved
        in a result file that indicated in the configuration.
        The name of the result saving file, which starts with the model name
        and followed by the given result file string as the affix.
    """

    def __init__(
        self,
        datasets,
        models,
        metrics=None,
        eval_scopes=None,
        model_dir=None,
        result_file=None,
        save_dir=None,
    ):
        """Initialise required inputs for the expriment pipeline."""
        self.datasets = datasets
        self.models = models
        self.metrics = metrics
        self.eval_scopes = eval_scopes
        self.result_file = result_file
        self.save_dir = save_dir
        self.update_config()

    def run(self):
        """Run the experiment."""
        results = []
        for data in self.datasets:
            for model in self.models:
                model.train(data)
                result = model.test(data.test[0])
                results.extend(result)
        print_result_as_table(results)

    def load_pretrained_model(self):
        """Load the pretrained model."""
        for data in self.datasets:
            for model in self.models:
                model.init_engine(data)
                model.load(model_dir=self.model_dir)
                model.predict(data.test[0])

    def update_config(self):
        """Update the configuration of models."""
        if self.metrics is not None:
            for model in self.models:
                model.config["system"]["metrics"] = self.metrics
        if self.eval_scopes is not None:
            for model in self.models:
                model.config["system"]["k"] = self.eval_scopes
        if self.result_file is not None:
            for idx, model in enumerate(self.models):
                model.config["system"]["result_file"] = (
                    "model_"
                    + str(idx)
                    + "_"
                    + self.config["model"]["model"]
                    + "_"
                    + self.result_file
                )
        if self.save_dir is not None:
            for model in self.models:
                model.config["system"]["result_dir"] = self.save_dir
