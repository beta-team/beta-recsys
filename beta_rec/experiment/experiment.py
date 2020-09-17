import os
from datetime import datetime
from beta_rec.recommenders.recommender import Recommender
from beta_rec.data import BaseData


class Experiment:
    """ The experiment class that enables the flow of an experiment with this beta-rec platform.  
    
    Args:
    dataset: :obj: '<beta_rec.datasets>', required 
            the experimental dataset (e.g. MovieLens)
    eval_methods: : array of string, required
            the evaluation method (e.g. ['load_leave_one_out'])
    models: array of :obj:`<beta_rec.recommenders>`, required
        A collection of recommender models to evaluate, e.g., [MF, GCN].
    metrics: array of string, required
        A collection of metrics to use to evaluate the recommender models, \
        e.g., [NDCG, MRR, Recall].
    model_dir: str, optional, default: None
        Path to a directory for loading a pretrained model
    save_dir: str, optional, default: None
        Path to a directory for storing trained models and logs. If None, 
        models will NOT be stored and logs will be saved in the current working directory.
    verbose: bool, optional, default: False
        print running logs to the terminal. If False, the print information will be hidden.
    """
    def __init__(
        self,
        dataset,
        eval_methods,
        models,
        metrics,
        model_dir=None,
        save_dir=None,
        verbose=False,
    ):
        self.dataset = dataset
        self.eval_method = eval_method
        self.models = self._validate_models(models)
        self.metrics = metrics
        self.verbose = verbose
        self.save_dir = save_dir
        self.config = {
            "config_file":"../configs/mf_default.json"
        }
        
    
    @staticmethod
    def _validate_models(input_models):
        if not hasattr(input_models, "__len__"):
            raise ValueError(
                "models have to be an array but {}".format(type(input_models))
            )

        valid_models = []
        for model in input_models:
            if isinstance(model, recommenders):
                valid_models.append(model)
        return valid_models
    
    def run(self):
        """Run the experiment"""
        import foo
        for eval_method in eval_methods:
            processed_dataset = getattr(self.dataset, eval_method)()
            data = BaseData(processed_dataset)
            for model in self.models:
                model.train(data)
                model.test(data.test[0])
    
    def load_pretrained_model(self):
        """Load the pretrained model"""
        for eval_method in eval_methods:
            processed_dataset = getattr(self.dataset, eval_method)()
            data = BaseData(processed_dataset)
            for model in self.models:
                model.init_engine(data) 
                model.load(model_dir = self.model_dir)
                model_score = model.predict(data.test[0])
                