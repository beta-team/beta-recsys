class Experiment:
    """ The experiment class that enables the flow of an experiment with
        this beta-rec platform.

    Args:
    datasets: array of :obj: '<beta_rec.datasets>', required
            the experimental datasets (e.g. MovieLens)
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
        models will NOT be stored and logs will be saved in the current
        working directory.
    verbose: bool, optional, default: False
        print running logs to the terminal. If False, the print information
        will be hidden.
    """

    def __init__(
        self, datasets, models, metrics, model_dir=None, save_dir=None, verbose=False,
    ):
        self.datasets = datasets
        self.models = models
        self.metrics = metrics
        self.verbose = verbose
        self.save_dir = save_dir
        self.config = {"config_file": "../configs/mf_default.json"}

    def run(self):
        """Run the experiment"""
        for data in self.datasets:
            for model in self.models:
                model.train(data)
                model.test(data.test[0])

    def load_pretrained_model(self):
        """Load the pretrained model"""
        for data in self.datasets:
            for model in self.models:
                model.init_engine(data)
                model.load(model_dir=self.model_dir)
                model.predict(data.test[0])
